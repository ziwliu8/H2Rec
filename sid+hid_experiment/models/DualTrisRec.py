import os
import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
from models.SASRec import SASRec_seq
from models.utils import Multi_CrossAttention, InfoNCE_Loss, Contrastive_Loss2
from models.RQVAEEmbedding import RQVAECodebookEmbedding
from models.PQVAEEmbedding import PQVAEEmbedding


logger = logging.getLogger(__name__)


class DualTrisRecSASRec(SASRec_seq):

    def __init__(self, user_num, item_num, device, args):
        from models.BaseModel import BaseSeqModel
        BaseSeqModel.__init__(self, user_num, item_num, device, args)
        self.args = args
        
        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = getattr(args, 'use_cross_att', True)
        
        self._init_hash_embedding(args)
        self._init_semantic_embedding(args)
        self._init_cross_attention(args)
        self._setup_infonce(args)
        self._setup_view_consistency(args)
        self._setup_backbones(device, args)
        self._init_common_modules(args)
        self._log_model_summary()

    def _init_hash_embedding(self, args):
        emb_path = os.path.join("data", args.dataset, "handled", "pca128_itm_emb_np.pkl")
        with open(emb_path, "rb") as f:
            id_item_emb = pickle.load(f)

        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        self.hash_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))
        self.hash_item_emb.weight.requires_grad = True

    def _resolve_semantic_paths(self, args, semantic_code_file):
        base_path = os.path.join("./data", args.dataset, "handled")
        default_path = os.path.join(args.dataset, semantic_code_file)

        if (
            args.dataset == "test_data"
            or args.dataset.startswith("./")
            or args.dataset.startswith("/")
            or not os.path.exists(os.path.join("./data", args.dataset))
        ):
            semantic_path = default_path
        else:
            semantic_path = os.path.join(base_path, semantic_code_file)

        codebook_file = semantic_code_file.replace(".code.", ".codebooks.").replace(".json", ".pkl")
        if semantic_path == default_path:
            codebook_path = os.path.join(args.dataset, codebook_file)
        else:
            codebook_path = os.path.join(base_path, codebook_file)

        return semantic_path, codebook_path

    def _init_semantic_embedding(self, args):
        semantic_code_file = getattr(
            args,
            "semantic_code_file",
            "yelp.code.rqvae_orig.4_128.emb1536.Cosine.recon.pca0.json"
        )
        semantic_path, codebook_path = self._resolve_semantic_paths(args, semantic_code_file)

        fusion_method = getattr(args, "semantic_fusion_method", "mean")
        learnable_codebooks = getattr(args, "learnable_codebooks", True)
        use_decoder = getattr(args, "use_decoder", False)
        freeze_decoder = getattr(args, "freeze_decoder", True)

        model_file = None
        if use_decoder:
            model_file = semantic_path.replace(".code.", ".model.").replace(".json", ".pth")

        quantizer_type = getattr(args, "semantic_encoder_type", "rqvae").lower()
        if quantizer_type == "rqvae":
            embedding_cls = RQVAECodebookEmbedding
        elif quantizer_type == "pqvae":
            embedding_cls = PQVAEEmbedding
        else:
            raise ValueError(f"Unsupported semantic encoder type: {quantizer_type}")

        embedding_kwargs = dict(
            codebook_file=codebook_path,
            semantic_code_file=semantic_path,
            embedding_dim=args.hidden_size,
            padding_idx=0,
            fusion_method=fusion_method,
            learnable=learnable_codebooks,
            device=self.dev,
        )

        if quantizer_type == "rqvae":
            embedding_kwargs.update(
                dict(
                    use_decoder=use_decoder,
                    model_file=model_file,
                    freeze_decoder=freeze_decoder,
                )
            )

        self.sid_item_emb = embedding_cls(**embedding_kwargs)

        self.n_sid_views = getattr(self.sid_item_emb, "n_codebooks", 1)
        self.use_dynamic_gate = getattr(args, "use_dynamic_gate", True)

    def _init_cross_attention(self, args):
        hidden_size = args.hidden_size
        if not self.use_cross_att:
            self.sid2hash_list = None
            self.sid_view_logits = None
            self.sid_gate_network = None
            self.hash2sid = None
            return

        self.sid2hash_list = nn.ModuleList(
            [Multi_CrossAttention(hidden_size, hidden_size, 4) for _ in range(self.n_sid_views)]
        )

        multi_view = self.n_sid_views > 1

        if not multi_view:
            self.use_dynamic_gate = False
            self.sid_gate_network = None
            self.sid_view_logits = None
            self.sid_hierarchy_bias = None
        elif self.use_dynamic_gate:
            self.sid_gate_network = nn.Sequential(
                nn.Linear(hidden_size + self.n_sid_views, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(hidden_size // 2, self.n_sid_views),
            )
            sid_hierarchy_bias = torch.linspace(1.0, 0.2, self.n_sid_views)
            self.register_buffer("sid_hierarchy_bias", sid_hierarchy_bias)
            self.sid_view_logits = None
        else:
            self.sid_gate_network = None
            self.sid_view_logits = nn.Parameter(torch.zeros(self.n_sid_views))
            self.sid_hierarchy_bias = None

        self.hash2sid = Multi_CrossAttention(hidden_size, hidden_size, 4)

    def _setup_infonce(self, args):
        self.use_infonce_loss = getattr(args, "use_infonce_loss", False)
        self.infonce_weight = getattr(args, "infonce_weight", 0.1)
        self.infonce_temperature = getattr(args, "infonce_temperature", 0.5)
        self.consecutive_window = getattr(args, "consecutive_window", 3)
        self.use_positive_weights = getattr(args, "use_positive_weights", True)
        self.use_dynamic_filter = getattr(args, "use_dynamic_filter", True)

        self.infonce_loss = None
        self.positive_filter = None
        self.hard_neg_sampler = None
        self.precomputed_positive_pairs = None
        self.precomputed_positive_weights = None
        self.positive_pairs_dense = None
        self.positive_pairs_file = getattr(args, "positive_pairs_file", None)

        if not self.use_infonce_loss:
            return

        use_hard_negatives = getattr(args, "use_hard_negatives", True)
        use_debias = getattr(args, "use_debias", True)

        self.infonce_loss = InfoNCE_Loss(
            temperature=self.infonce_temperature,
            use_hard_negatives=use_hard_negatives,
            debias=use_debias,
        )

        if self.positive_pairs_file:
            self._load_precomputed_positive_pairs(args)

        if self.use_dynamic_filter:
            from models.utils import DynamicPositiveFilter

            self.positive_filter = DynamicPositiveFilter(
                top_k_ratio=getattr(args, "positive_filter_ratio", 0.6),
                min_similarity=getattr(args, "positive_min_sim", 0.1),
            )

        if use_hard_negatives:
            from models.utils import HardNegativeSampler

            self.hard_neg_sampler = HardNegativeSampler(
                item_embeddings=self.hash_item_emb,
                n_hard_negatives=getattr(args, "n_hard_negatives", 5),
                temperature=0.5,
            )

    def _setup_view_consistency(self, args):
        """设置视角一致性对比学习（View Consistency Contrastive Learning）"""
        self.use_view_cl = getattr(args, "use_view_cl", False)
        self.view_cl_weight = getattr(args, "view_cl_weight", 0.1)
        self.view_cl_mask_ratio = getattr(args, "view_cl_mask_ratio", 0.5) 
        self.view_cl_mask_strategy = getattr(args, "view_cl_mask_strategy", "low_weight") 
        
        if not self.use_view_cl:
            self.view_mask_embeddings = None
            self.view_cl_loss = None
            return
    
        self.view_mask_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(args.hidden_size) * 0.02)
            for _ in range(self.n_sid_views)
        ])
        
        from models.utils import Contrastive_Loss2
        self.view_cl_loss = Contrastive_Loss2(tau=getattr(args, "view_cl_temperature", 1.0))

    def _setup_backbones(self, device, args):
        from models.SASRec import SASRecBackbone

        self.hash_backbone = SASRecBackbone(device, args)
        self.share_backbone = getattr(args, "share_backbone", True)
        self.sid_backbone = self.hash_backbone if self.share_backbone else SASRecBackbone(device, args)
        #self.sid_backbone = SASRecBackbone(device, args)

    def _init_common_modules(self, args):
        self.pos_emb = torch.nn.Embedding(args.max_len + 100, args.hidden_size)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.filter_init_modules = ["sid_item_emb", "hash_item_emb"]
        self._init_weights()

    def _log_model_summary(self):
        logger.debug(
            "[DualTrisRec] init | cross_att=%s | views=%d | shared_backbone=%s",
            self.use_cross_att,
            getattr(self, "n_sid_views", 1),
            getattr(self, "share_backbone", True),
        )
        if self.use_infonce_loss:
            logger.debug(
                "[DualTrisRec] InfoNCE enabled | weight=%.3f | temperature=%.3f | positives=%s",
                self.infonce_weight,
                self.infonce_temperature,
                self.positive_pairs_file or "runtime",
            )
        if self.use_view_cl:
            logger.debug(
                "[DualTrisRec] ViewCL enabled | weight=%.3f | mask_ratio=%.2f | strategy=%s",
                self.view_cl_weight,
                self.view_cl_mask_ratio,
                self.view_cl_mask_strategy,
            )

    @staticmethod
    def _scaled_positional_dropout(tensor, pos_embed, dropout_layer, scale_dim):
        if pos_embed is None:
            raise ValueError("Position embedding must be precomputed.")
        return dropout_layer(tensor * (scale_dim ** 0.5) + pos_embed)

    def _get_hash_embedding(self, log_seqs):
        """获取hash ID embedding"""
        return self.hash_item_emb(log_seqs)
    
    def _get_semantic_embedding(self, log_seqs):
        """获取semantic ID embedding"""
        return self.sid_item_emb(log_seqs)
    
    def _get_embedding(self, log_seqs):
        """获取联合embedding：拼接sid和hash"""
        hash_emb = self._get_hash_embedding(log_seqs)
        sid_emb = self._get_semantic_embedding(log_seqs)
        
        # 在最后一维拼接
        combined_emb = torch.cat([hash_emb, sid_emb], dim=-1)
        return combined_emb
    
    def _load_precomputed_positive_pairs(self, args):
        """加载预计算的正样本对矩阵和权重"""
        import pickle
        
        # 处理路径
        if args.dataset.startswith('./') or args.dataset.startswith('/'):
            pairs_path = os.path.join(args.dataset, self.positive_pairs_file)
        else:
            pairs_path = os.path.join('./data', args.dataset, 'handled', self.positive_pairs_file)
        
        logger.debug("[DualTrisRec] 加载预计算正样本对: %s", pairs_path)
        
        try:
            # 加载正样本对
            with open(pairs_path, 'rb') as f:
                self.precomputed_positive_pairs = pickle.load(f)
            
            n_items_with_pairs = len(self.precomputed_positive_pairs)
            total_pairs = sum(len(v) for v in self.precomputed_positive_pairs.values())
            logger.debug("  成功加载正样本对: %d 个物品，总计 %d 对",
                         n_items_with_pairs, total_pairs)
            
            if self.use_positive_weights:
                weights_path = pairs_path.replace('.pkl', '_weights.pkl')
                try:
                    with open(weights_path, 'rb') as f:
                        self.precomputed_positive_weights = pickle.load(f)
                    logger.debug("  成功加载权重矩阵: %s", weights_path)
                    

                    all_weights = []
                    for item_weights in self.precomputed_positive_weights.values():
                        all_weights.extend(item_weights.values())
                    if all_weights:
                        logger.debug("    权重范围: [%.3f, %.3f]",
                                     min(all_weights), max(all_weights))
                        logger.debug("    平均权重: %.3f",
                                     sum(all_weights) / len(all_weights))
                except FileNotFoundError:
                    logger.info("  未找到权重文件，使用等权")
                    self.precomputed_positive_weights = None
            
        except FileNotFoundError:
            logger.warning("正样本对文件未找到，使用简化的对比学习")
            logger.warning("建议运行: python precompute_positive_pairs_v2.py --dataset %s", args.dataset)
            self.precomputed_positive_pairs = None
            self.precomputed_positive_weights = None
        
        # 将字典转换为GPU密集张量（性能优化V2）
        if self.precomputed_positive_pairs is not None:
            self._convert_pairs_to_dense_tensor()
    
    def _convert_pairs_to_dense_tensor(self):

        logger.debug("[DualTrisRec] 转换正样本对为GPU密集张量")
        
        row_indices = []
        col_indices = []
        weights = []
        
        # 收集所有item-item对和权重
        for item_i, pos_items in self.precomputed_positive_pairs.items():
            for pos_item in pos_items:
                row_indices.append(item_i)
                col_indices.append(pos_item)
                
                # 获取权重（如果有）
                if self.precomputed_positive_weights and item_i in self.precomputed_positive_weights:
                    weight = self.precomputed_positive_weights[item_i].get(pos_item, 0.5)
                else:
                    weight = 0.5  # 默认权重
                weights.append(weight)
        
        if row_indices:
            indices = torch.LongTensor([row_indices, col_indices])
            values = torch.FloatTensor(weights)
            size = (self.item_num + 2, self.item_num + 2)  # +2 for padding and mask tokens
            
            sparse_tensor = torch.sparse_coo_tensor(
                indices, values, size, device='cpu'
            ).coalesce()  # coalesce合并重复索引
            
            self.positive_pairs_dense = sparse_tensor.to_dense().to(self.dev)
            
            logger.debug("  正样本对矩阵大小: %s", size)
            logger.debug("  非零元素数: %d", len(row_indices))
            logger.debug("  稀疏度比例: %.4f%%",
                         len(row_indices) / (size[0] * size[1]) * 100)
            logger.debug("  存储格式: GPU密集张量")
            memory_mb = (size[0] * size[1] * 4) / (1024 * 1024)
            logger.debug("  估计内存占用: %.1f MB", memory_mb)
        else:
            self.positive_pairs_dense = None
            logger.warning("未发现有效的正样本对")
    
    def _build_positive_mask(self, item_ids):
        batch_size = item_ids.size(0)
        device = item_ids.device
        
        # 初始化：对角线为1（同一item的sid和hash是正样本对）
        positive_mask = torch.eye(batch_size, device=device)
        positive_weights = torch.eye(batch_size, device=device) 
        

        if self.positive_pairs_dense is not None:
            positive_weights_batch = self.positive_pairs_dense[item_ids][:, item_ids]
            
            # 构建mask（权重>0的位置）
            positive_mask_batch = (positive_weights_batch > 0).float()
            
            # 合并到对角线mask（取并集）
            positive_mask = torch.max(positive_mask, positive_mask_batch)
            
            # 合并权重（优先使用预计算权重，对角线保持1.0）
            positive_weights = torch.where(
                positive_mask_batch > 0,
                positive_weights_batch,
                positive_weights
            )
            positive_weights.fill_diagonal_(1.0)
        
        return positive_mask, positive_weights
    
    def _build_positive_mask_from_sequence(self, seq, item_ids):
        return self._build_positive_mask(item_ids)

    def log2feats(self, log_seqs, positions, return_view_cl_data=False):
    
        positions = positions.long()
        pos_embed = self.pos_emb(positions)

        hash_seqs = self._get_hash_embedding(log_seqs)
        # 多视角SID：每个码本一个视角
        sid_views = None
        if hasattr(self.sid_item_emb, 'get_per_view_embeddings'):
            sid_views = self.sid_item_emb.get_per_view_embeddings(log_seqs, project_to_dim=self.hash_item_emb.embedding_dim)
        else:
            # 回退到单视角融合
            sid_views = [self._get_semantic_embedding(log_seqs)]
        
        hash_seqs = self._scaled_positional_dropout(
            hash_seqs,
            pos_embed,
            self.emb_dropout,
            self.hash_item_emb.embedding_dim,
        )
        
        processed_sid_views = []
        for v in sid_views:
            processed_sid_views.append(
                self._scaled_positional_dropout(
                    v,
                    pos_embed,
                    self.emb_dropout,
                    self.hash_item_emb.embedding_dim,
                )
            )
        
        # === 路径1: 多视角SID增强Hash ===
        view_cl_data = None
        alphas = None
        
        if self.use_cross_att:
            enhanced_views = []
            for i, vv in enumerate(processed_sid_views):
                attn_mod = self.sid2hash_list[i if i < len(self.sid2hash_list) else -1]
                enhanced = attn_mod(vv, hash_seqs, log_seqs)
                enhanced_views.append(enhanced)
            
            if len(enhanced_views) > 1:
                if self.use_dynamic_gate and self.sid_gate_network is not None:
                    seq_lens = (log_seqs != 0).sum(dim=1)  # (bs,)
                    batch_indices = torch.arange(hash_seqs.size(0), device=hash_seqs.device)
                    last_valid_idx = (seq_lens - 1).clamp(min=0)
                    seq_repr = hash_seqs[batch_indices, last_valid_idx, :]  # (bs, hidden_size)
                    
                    hierarchy_enc = self.sid_hierarchy_bias.unsqueeze(0).expand(seq_repr.size(0), -1)  # (bs, n_views)
                    
                    gate_input = torch.cat([seq_repr, hierarchy_enc], dim=-1)  # (bs, hidden_size + n_views)
                    gate_logits = self.sid_gate_network(gate_input)  # (bs, n_views)
                    
                    gate_logits = gate_logits + self.sid_hierarchy_bias.unsqueeze(0)
                    
                    alphas = torch.softmax(gate_logits, dim=-1)  # (bs, n_views)
                    
                    cross_hash_seqs = torch.zeros_like(enhanced_views[0])
                    for i, enh in enumerate(enhanced_views):
                        # alphas[:, i]: (bs,) -> (bs, 1, 1)
                        weight = alphas[:, i].unsqueeze(-1).unsqueeze(-1)
                        cross_hash_seqs = cross_hash_seqs + weight * enh
                    
                    self._last_mv = {
                        'sid_views': sid_views,
                        'hash_seqs': hash_seqs,
                        'alphas': alphas.mean(dim=0).detach()  # (n_views,)
                    }
                elif hasattr(self, 'sid_view_logits') and self.sid_view_logits is not None:
                    alphas = torch.softmax(self.sid_view_logits, dim=0)
                    cross_hash_seqs = 0
                    for a, enh in zip(alphas, enhanced_views):
                        cross_hash_seqs = cross_hash_seqs + a * enh
                    self._last_mv = {
                        'sid_views': sid_views,
                        'hash_seqs': hash_seqs,
                        'alphas': alphas.detach()
                    }
                else:
                    alphas = torch.ones(len(enhanced_views), device=hash_seqs.device) / len(enhanced_views)
                    cross_hash_seqs = sum(enhanced_views) / len(enhanced_views)
                    self._last_mv = {
                        'sid_views': sid_views,
                        'hash_seqs': hash_seqs,
                        'alphas': None
                    }
            else:
                cross_hash_seqs = enhanced_views[0]
                self._last_mv = {
                    'sid_views': sid_views,
                    'hash_seqs': hash_seqs,
                    'alphas': None
                }
        else:
            cross_hash_seqs = hash_seqs
            self._last_mv = None
        
        # 路径1：通过hash backbone（经过SID增强的Hash）
        enhanced_hash_feats = self.hash_backbone(cross_hash_seqs, log_seqs)
        
        # === 路径2: Hash增强SID（对称设计）===
        sid_seqs = self._scaled_positional_dropout(
            self._get_semantic_embedding(log_seqs),
            pos_embed,
            self.emb_dropout,
            self.hash_item_emb.embedding_dim,
        )
        
        # Hash增强SID（通过cross attention）
        if self.use_cross_att and self.hash2sid is not None:
            # Hash作为query增强SID
            cross_sid_seqs = self.hash2sid(hash_seqs, sid_seqs, log_seqs)
        else:
            # 如果不使用cross attention，使用纯SID
            cross_sid_seqs = sid_seqs
        
        # 通过SID backbone
        enhanced_sid_feats = self.sid_backbone(cross_sid_seqs, log_seqs)
        
        # 拼接两路增强特征
        log_feats = torch.cat([enhanced_hash_feats, enhanced_sid_feats], dim=-1)
        
        # === 视角一致性对比学习（View Consistency CL）===
        if return_view_cl_data and self.use_view_cl and self.training and alphas is not None and len(processed_sid_views) > 1:
            batch_size, seq_len, hidden_dim = processed_sid_views[0].shape
            n_views = len(processed_sid_views)
            

            n_mask = max(1, int(n_views * self.view_cl_mask_ratio))
            
            if self.view_cl_mask_strategy == "low_weight":   
                if alphas.dim() == 2:
                    _, mask_indices = torch.topk(alphas, n_mask, dim=-1, largest=False)  # (bs, n_mask)
                else:
                    _, mask_indices = torch.topk(alphas, n_mask, largest=False)  # (n_mask,)
                    mask_indices = mask_indices.unsqueeze(0).expand(batch_size, -1)  # (bs, n_mask)
            elif self.view_cl_mask_strategy == "high_weight":
                if alphas.dim() == 2:
                    _, mask_indices = torch.topk(alphas, n_mask, dim=-1, largest=True)
                else:
                    _, mask_indices = torch.topk(alphas, n_mask, largest=True)
                    mask_indices = mask_indices.unsqueeze(0).expand(batch_size, -1)
            else:  # random
                mask_indices = torch.stack([
                    torch.randperm(n_views, device=log_seqs.device)[:n_mask]
                    for _ in range(batch_size)
                ], dim=0)  # (bs, n_mask)
            
            masked_sid_views = []
            for view_idx in range(n_views):
                view_emb = processed_sid_views[view_idx].clone()  # (bs, seq_len, hidden_dim)
                should_mask = (mask_indices == view_idx).any(dim=-1)  # (bs,)
                
                if should_mask.any():
                    mask_emb = self.view_mask_embeddings[view_idx]  # (hidden_dim,)
                    mask_emb_expanded = mask_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
                    should_mask_expanded = should_mask.unsqueeze(-1).unsqueeze(-1).expand_as(view_emb)
                    view_emb = torch.where(should_mask_expanded, mask_emb_expanded, view_emb)
                
                masked_sid_views.append(view_emb)
            
            if alphas.dim() == 2:
                masked_fused = torch.zeros_like(masked_sid_views[0])
                for i, mv in enumerate(masked_sid_views):
                    weight = alphas[:, i].unsqueeze(-1).unsqueeze(-1)  # (bs, 1, 1)
                    masked_fused = masked_fused + weight * mv
            else:
                masked_fused = sum(a * mv for a, mv in zip(alphas, masked_sid_views))
            
            if alphas.dim() == 2:
                full_fused = torch.zeros_like(processed_sid_views[0])
                for i, pv in enumerate(processed_sid_views):
                    weight = alphas[:, i].unsqueeze(-1).unsqueeze(-1)
                    full_fused = full_fused + weight * pv
            else:
                full_fused = sum(a * pv for a, pv in zip(alphas, processed_sid_views))
            
            # 通过sid backbone获取用户表征
            u1_seq = self.sid_backbone(masked_fused, log_seqs)  # masked表征
            u2_seq = self.sid_backbone(full_fused, log_seqs)    # 完整表征
            
            # 取最后一个有效位置的表征
            seq_lens = (log_seqs != 0).sum(dim=1)
            batch_indices = torch.arange(batch_size, device=log_seqs.device)
            last_valid_idx = (seq_lens - 1).clamp(min=0)
            
            u1 = u1_seq[batch_indices, last_valid_idx, :]  # (bs, hidden_dim)
            u2 = u2_seq[batch_indices, last_valid_idx, :]  # (bs, hidden_dim)
            
            view_cl_data = {
                'u_masked': u1,
                'u_full': u2,
                'mask_indices': mask_indices,
                'alphas': alphas.detach() if alphas.dim() == 2 else alphas
            }
        
        # 根据请求返回不同的数据
        if return_view_cl_data:
            return log_feats, view_cl_data
        
        return log_feats

    def forward(self, seq, pos, neg, positions, **kwargs):
        return_view_cl = self.use_view_cl and self.training
        
        if return_view_cl:
            log_feats, view_cl_data = self.log2feats(
                seq, positions, return_view_cl_data=True
            )
        else:
            log_feats = self.log2feats(seq, positions)
            view_cl_data = None
        
        log_feats = log_feats[:, -1, :].unsqueeze(1)  # 取最后一个token
        
        pos_embs = self._get_embedding(pos.unsqueeze(1))
        neg_embs = self._get_embedding(neg)
        
        # 计算logits
        pos_logits = torch.mul(log_feats, pos_embs).sum(dim=-1)
        neg_logits = torch.mul(log_feats, neg_embs).sum(dim=-1)
        
        # 计算损失
        pos_labels = torch.ones(pos_logits.shape, device=self.dev)
        neg_labels = torch.zeros(neg_logits.shape, device=self.dev)
        
        indices = (pos != 0)
        pos_loss = self.loss_func(pos_logits[indices], pos_labels[indices])
        neg_loss = self.loss_func(neg_logits[indices], neg_labels[indices])
        main_loss = pos_loss + neg_loss
        
        if self.use_infonce_loss and self.training:
            batch_hash_emb = self._get_hash_embedding(pos)  # [batch_size, hidden_size]
            batch_sid_emb = self._get_semantic_embedding(pos)  # [batch_size, hidden_size]
            positive_mask, positive_weights = self._build_positive_mask_from_sequence(seq, pos)
            
            if self.use_dynamic_filter and hasattr(self, 'positive_filter'):
                seq_repr = log_feats[:, -1, :]  
                batch_hash_emb_2d = batch_hash_emb.unsqueeze(1).expand(-1, positive_mask.size(1), -1)
                filtered_mask, filtered_weights = self.positive_filter(
                    seq_repr, batch_hash_emb_2d, positive_mask
                )
                if self.use_positive_weights:
                    positive_weights = filtered_weights * positive_weights
                else:
                    positive_weights = filtered_mask
            elif not self.use_positive_weights:
                positive_weights = positive_mask
            
            hard_negatives = None
            if hasattr(self, 'hard_neg_sampler'):
                try:
                    hard_neg_ids = self.hard_neg_sampler(pos, seq)  # [batch_size, n_hard_neg]
                    hard_negatives = self._get_hash_embedding(hard_neg_ids)  # [batch_size, n_hard_neg, dim]
                except:
                    pass  
            
            infonce_loss_s2h = self.infonce_loss(
                batch_sid_emb, batch_hash_emb,
                positive_mask=positive_mask,
                positive_weights=positive_weights,
                hard_negatives=hard_negatives
            )
            infonce_loss_h2s = self.infonce_loss(
                batch_hash_emb, batch_sid_emb,
                positive_mask=positive_mask,
                positive_weights=positive_weights,
                hard_negatives=hard_negatives
            )
            infonce_loss = (infonce_loss_s2h + infonce_loss_h2s) / 2.0
            
            main_loss = main_loss + self.infonce_weight * infonce_loss
        
        # === 视角一致性对比学习损失：防止过度依赖主导粒度 ===
        if self.use_view_cl and self.training and view_cl_data is not None:
            u_masked = view_cl_data['u_masked']  # (bs, hidden_dim)
            u_full = view_cl_data['u_full']      # (bs, hidden_dim)
            
            view_cl_loss = self.view_cl_loss(u_masked, u_full)
            
            main_loss = main_loss + self.view_cl_weight * view_cl_loss
        
        return main_loss

    def predict(self, seq, item_indices, positions, **kwargs):

        log_feats = self.log2feats(seq, positions)
        final_feat = log_feats[:, -1, :]  # 取最后一个token
        
        item_embs = self._get_embedding(item_indices)
        
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        
        return logits

    def get_user_emb(self, seq, positions, **kwargs):
        """获取用户embedding"""
        log_feats = self.log2feats(seq, positions)
        final_feat = log_feats[:, -1, :]
        
        return final_feat
    
    def get_parameter_groups(self, 
                           hash_lr=None,
                           semantic_codebook_lr=None, 
                           semantic_projection_lr=None, 
                           semantic_decoder_lr=None,
                           hash_backbone_lr=None,
                           sid_backbone_lr=None,
                           backbone_lr=None):
        param_groups = []
        
        # Hash embedding参数
        if hash_lr is not None:
            param_groups.append({
                'params': self.hash_item_emb.parameters(),
                'lr': hash_lr,
                'name': 'hash_embedding'
            })
        
        # Semantic embedding参数
        if hasattr(self.sid_item_emb, 'get_codebook_parameters'):
            codebook_params = self.sid_item_emb.get_codebook_parameters()
            if codebook_params and semantic_codebook_lr is not None:
                param_groups.append({
                    'params': codebook_params,
                    'lr': semantic_codebook_lr,
                    'name': 'semantic_codebook'
                })
        
        if hasattr(self.sid_item_emb, 'get_projection_parameters'):
            projection_params = self.sid_item_emb.get_projection_parameters()
            if projection_params and semantic_projection_lr is not None:
                param_groups.append({
                    'params': projection_params,
                    'lr': semantic_projection_lr,
                    'name': 'semantic_projection'
                })
        
        if hasattr(self.sid_item_emb, 'get_decoder_parameters'):
            decoder_params = self.sid_item_emb.get_decoder_parameters()
            if decoder_params and semantic_decoder_lr is not None:
                param_groups.append({
                    'params': decoder_params,
                    'lr': semantic_decoder_lr,
                    'name': 'semantic_decoder'
                })
        
        # Hash Backbone参数
        if hash_backbone_lr is not None or backbone_lr is not None:
            lr = hash_backbone_lr if hash_backbone_lr is not None else backbone_lr
            param_groups.append({
                'params': self.hash_backbone.parameters(),
                'lr': lr,
                'name': 'hash_backbone'
            })
        
        # SID Backbone参数（如果不共享）
        if not self.share_backbone:
            if sid_backbone_lr is not None or backbone_lr is not None:
                lr = sid_backbone_lr if sid_backbone_lr is not None else backbone_lr
                param_groups.append({
                    'params': self.sid_backbone.parameters(),
                    'lr': lr,
                    'name': 'sid_backbone'
                })
        
        # 其他参数（cross attention、门控等）
        other_params = []
        for name, param in self.named_parameters():
            skip = False
            
            # 跳过已经在其他组的参数
            if 'hash_item_emb' in name and hash_lr is not None:
                skip = True
            if 'sid_item_emb.codebook_embeddings' in name and semantic_codebook_lr is not None:
                skip = True
            if any(proj_name in name for proj_name in ['sid_item_emb.projection', 'sid_item_emb.projections', 'sid_item_emb.final_projection', 'sid_item_emb.decoder_input_projection', 'sid_item_emb.decoder_output_projection']) and semantic_projection_lr is not None:
                skip = True
            if 'sid_item_emb.decoder' in name and semantic_decoder_lr is not None:
                skip = True
            if 'hash_backbone' in name and (hash_backbone_lr is not None or backbone_lr is not None):
                skip = True
            if not self.share_backbone and 'sid_backbone' in name and (sid_backbone_lr is not None or backbone_lr is not None):
                skip = True
            
            if not skip:
                other_params.append(param)
        
        if other_params:
            lr = backbone_lr if backbone_lr is not None else 1e-3
            param_groups.append({
                'params': other_params,
                'lr': lr,
                'name': 'others'
            })
        
        return param_groups
    
    def save_semantic_codebooks(self, save_path: str):
        """保存更新后的semantic码本"""
        if hasattr(self.sid_item_emb, 'save_updated_codebooks'):
            self.sid_item_emb.save_updated_codebooks(save_path)
            logger.info("[DualTrisRec] Semantic码本已保存到: %s", save_path)
        else:
            logger.info("[DualTrisRec] 当前semantic embedding层不支持码本保存")


class DualTrisRecSASRec_seq(DualTrisRecSASRec):

    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)

    def forward(self, seq, pos, neg, positions, **kwargs):

        return_view_cl = self.use_view_cl and self.training
        
        if return_view_cl:
            log_feats, view_cl_data = self.log2feats(
                seq, positions, return_view_cl_data=True
            )
        else:
            log_feats = self.log2feats(seq, positions)
            view_cl_data = None
        
        # (bs, max_seq_len, hidden_size)
        
        # 获取正负样本的hash embedding（不再拼接）
        pos_embs = self._get_embedding(pos)  # (bs, max_seq_len, hidden_size)
        neg_embs = self._get_embedding(neg)  # (bs, max_seq_len, hidden_size)

        # 计算logits
        pos_logits = (log_feats * pos_embs).sum(dim=-1)  # (bs, max_seq_len)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)  # (bs, max_seq_len)

        # 计算损失
        pos_labels = torch.ones(pos_logits.shape, device=self.dev)
        neg_labels = torch.zeros(neg_logits.shape, device=self.dev)
        
        indices = (pos != 0)  # 不计算padding位置的损失
        pos_loss = self.loss_func(pos_logits[indices], pos_labels[indices])
        neg_loss = self.loss_func(neg_logits[indices], neg_labels[indices])
        loss = pos_loss + neg_loss
        
        # InfoNCE对比学习损失：拉近sid和hash表征（序列版本，升级版）
        if hasattr(self, 'use_infonce_loss') and self.use_infonce_loss and self.training:
            # 收集所有有效位置的item embeddings
            valid_pos = pos[indices]  # 所有有效的正样本item ID
            
            if valid_pos.numel() > 0 and valid_pos.size(0) > 1:  # 至少需要2个样本
                # 获取这些item的hash和sid embedding
                batch_hash_emb = self._get_hash_embedding(valid_pos)  # [N, hidden_size]
                batch_sid_emb = self._get_semantic_embedding(valid_pos)  # [N, hidden_size]
                
                # 构建正样本对矩阵和权重
                positive_mask, positive_weights = self._build_positive_mask(valid_pos)
                
                # 使用权重（如果启用）
                if not self.use_positive_weights:
                    positive_weights = positive_mask
                
                # 硬负样本（序列版本不使用，因为样本已经很丰富）
                hard_negatives = None
                
                # 双向对比损失（使用权重）
                infonce_loss_s2h = self.infonce_loss(
                    batch_sid_emb, batch_hash_emb,
                    positive_mask=positive_mask,
                    positive_weights=positive_weights,
                    hard_negatives=hard_negatives
                )
                infonce_loss_h2s = self.infonce_loss(
                    batch_hash_emb, batch_sid_emb,
                    positive_mask=positive_mask,
                    positive_weights=positive_weights,
                    hard_negatives=hard_negatives
                )
                infonce_loss = (infonce_loss_s2h + infonce_loss_h2s) / 2.0
                
                loss = loss + self.infonce_weight * infonce_loss
        
        # === 视角一致性对比学习损失（序列版本）===
        if self.use_view_cl and self.training and view_cl_data is not None:
            u_masked = view_cl_data['u_masked']  # (bs, hidden_dim)
            u_full = view_cl_data['u_full']      # (bs, hidden_dim)
            
            # 使用Contrastive_Loss2拉近masked和full表征的距离
            view_cl_loss = self.view_cl_loss(u_masked, u_full)
            
            loss = loss + self.view_cl_weight * view_cl_loss

        return loss
