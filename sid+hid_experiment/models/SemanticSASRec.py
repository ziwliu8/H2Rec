import numpy as np
import os
import torch
import torch.nn as nn
import pickle
import json
from models.BaseModel import BaseSeqModel
from models.utils import PointWiseFeedForward
from models.RQVAEEmbedding import RQVAECodebookEmbedding


class CodebookEmbedding(nn.Module):
    
    def __init__(self, codebooks, semantic_codes, embedding_dim, padding_idx=0, scale_factor=False):
        super(CodebookEmbedding, self).__init__()
        
        self.codebooks = codebooks
        self.semantic_codes = semantic_codes
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.n_codebooks = len(codebooks)
        self.num_items = len(semantic_codes)
        self.scale_factor = scale_factor
        
        self.codebook_embeddings = nn.ModuleList()
        for i, codebook in enumerate(codebooks):
            codebook_tensor = torch.from_numpy(codebook).float()
            embedding_layer = nn.Embedding.from_pretrained(codebook_tensor, freeze=False, padding_idx=None)
            self.codebook_embeddings.append(embedding_layer)
        
        self._build_semantic_lookup_table()
        
        codebook_dim = codebooks[0].shape[1] if codebooks else embedding_dim
        if codebook_dim != embedding_dim:
            self.projection = nn.Linear(codebook_dim, embedding_dim)
            self.codebook_dim = codebook_dim
        else:
            self.projection = None
            self.codebook_dim = embedding_dim
        
        self.code_dims = [cb.shape[1] for cb in codebooks]
        self.total_dim = sum(self.code_dims)
        
    def _build_semantic_lookup_table(self):
        semantic_table = torch.zeros(self.num_items, self.n_codebooks, dtype=torch.long)
        
        for item_id, semantic_code in enumerate(self.semantic_codes):
            if semantic_code:
                semantic_table[item_id] = torch.tensor(semantic_code, dtype=torch.long)
        
        self.register_buffer('semantic_table', semantic_table)
    
    def forward(self, item_ids):
        original_shape = item_ids.shape
        item_ids_flat = item_ids.view(-1)
        batch_size = item_ids_flat.size(0)
        
        item_ids_safe = torch.clamp(item_ids_flat, 0, self.num_items - 1)
        
        semantic_codes = self.semantic_table[item_ids_safe]
        
        pooled_embeddings = None
        
        for codebook_idx in range(self.n_codebooks):
            codes = semantic_codes[:, codebook_idx]
            
            code_embs = self.codebook_embeddings[codebook_idx](codes)
            
            if pooled_embeddings is None:
                pooled_embeddings = code_embs
            else:
                pooled_embeddings = pooled_embeddings + code_embs
        
        if hasattr(self, 'scale_factor') and self.scale_factor:
            pooled_embeddings = pooled_embeddings / self.n_codebooks
        
        padding_mask = (item_ids_flat == self.padding_idx).unsqueeze(-1)
        pooled_embeddings = pooled_embeddings * (~padding_mask).float()
        
        if self.projection is not None:
            final_embeddings = self.projection(pooled_embeddings)
        else:
            final_embeddings = pooled_embeddings
        
        if len(original_shape) == 1:
            return final_embeddings
        else:
            return final_embeddings.view(original_shape[0], original_shape[1], -1)


class SemanticSASRecBackbone(nn.Module):

    def __init__(self, device, args) -> None:
        super().__init__()

        self.dev = device
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)

        for _ in range(args.trm_num):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_size,
                                                        args.num_heads,
                                                        args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_size, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, seqs, log_seqs):
        timeline_mask = (log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats


class SemanticSASRec(BaseSeqModel):
    
    def __init__(self, user_num, item_num, device, args):
        super(SemanticSASRec, self).__init__(user_num, item_num, device, args)

        self.use_semantic_id = getattr(args, 'use_semantic_id', False)
        
        if self.use_semantic_id:
            self._load_semantic_components(args)
            
            if hasattr(self, 'use_rqvae_embedding') and self.use_rqvae_embedding:
                fusion_method = getattr(args, 'fusion_method', 'sum')
                if hasattr(args, 'freeze_codebooks') and args.freeze_codebooks:
                    learnable_codebooks = False
                else:
                    learnable_codebooks = getattr(args, 'learnable_codebooks', True)
                use_decoder = getattr(args, 'use_decoder', False)
                freeze_decoder = getattr(args, 'freeze_decoder', True)
                
                model_file = None
                if use_decoder:
                    model_file = self.semantic_path.replace('.code.', '.model.').replace('.json', '.pth')
                
                self.item_emb = RQVAECodebookEmbedding(
                    codebook_file=self.codebook_path,
                    semantic_code_file=self.semantic_path,
                    embedding_dim=args.hidden_size,
                    padding_idx=0,
                    fusion_method=fusion_method,
                    learnable=learnable_codebooks,
                    device=self.dev,
                    use_decoder=use_decoder,
                    model_file=model_file,
                    freeze_decoder=freeze_decoder
                )
                print(f"[SemanticSASRec] 使用RQVAE embedding，融合方法: {fusion_method}, 可学习: {learnable_codebooks}, decoder: {use_decoder}")
            else:
                self.item_emb = CodebookEmbedding(
                    self.codebooks, 
                    self.semantic_codes, 
                    args.hidden_size, 
                    padding_idx=0
                )
        else:
            self.item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)

        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.backbone = SemanticSASRecBackbone(device, args)
        self.loss_func = torch.nn.BCEWithLogitsLoss()

        self._init_weights()

    def _load_semantic_components(self, args):
        dataset = args.dataset
        
        semantic_code_file = getattr(args, 'semantic_code_file', 'beauty.code.fixed_rq.12_256.pca128.json')
        semantic_path = os.path.join('./data', dataset, 'handled', semantic_code_file)
        
        if 'rqvae_orig' in semantic_code_file:
            codebook_file = semantic_code_file.replace('.code.', '.codebooks.').replace('.json', '.pkl')
            codebook_path = os.path.join('./data', dataset, 'handled', codebook_file)
            
            print(f"[SemanticSASRec] 使用RQVAE框架格式")
            print(f"  - 语义代码文件: {semantic_path}")
            print(f"  - 码本文件: {codebook_path}")
            
            self.use_rqvae_embedding = True
            self.semantic_path = semantic_path
            self.codebook_path = codebook_path
            
        elif 'direct_rq' in semantic_code_file:
            model_file = semantic_code_file.replace('.code.', '.model.').replace('.json', '.pth')
            model_path = os.path.join('./data', dataset, 'handled', model_file)
            
            print(f"Loading codebooks from DirectRQ model: {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')
            model_state = checkpoint['model_state_dict']
            config = checkpoint['config']
            
            with open(semantic_path, 'r') as f:
                self.semantic_codes = json.load(f)
            
            self.codebooks = []
            n_layers = config['n_layers']
            for layer_idx in range(n_layers):
                codebook_key = f'quantizer.codebooks.{layer_idx}.weight'
                if codebook_key in model_state:
                    codebook_weight = model_state[codebook_key].numpy()
                    self.codebooks.append(codebook_weight)
                else:
                    raise KeyError(f"Codebook layer {layer_idx} not found in model state")
            
            print(f"Loaded {len(self.codebooks)} codebooks from DirectRQ model")
            print(f"Codebook config: n_layers={config['n_layers']}, n_clusters={config['n_clusters']}, embedding_dim={config['embedding_dim']}")
            self.use_rqvae_embedding = False
            
        else:
            codebook_file = semantic_code_file.replace('.json', '.pkl').replace('.code.', '.codebooks.')
            codebook_path = os.path.join('./data', dataset, 'handled', codebook_file)
            
            print(f"Loading codebooks from pickle file: {codebook_path}")
            with open(semantic_path, 'r') as f:
                self.semantic_codes = json.load(f)
            with open(codebook_path, 'rb') as f:
                self.codebooks = pickle.load(f)
            
            print(f"Loaded {len(self.codebooks)} codebooks and {len(self.semantic_codes)} semantic codes")
            self.use_rqvae_embedding = False

    def _get_embedding(self, log_seqs):
        if self.use_semantic_id:
            item_seq_emb = self.item_emb(log_seqs)
        else:
            item_seq_emb = self.item_emb(log_seqs)
        
        return item_seq_emb

    def log2feats(self, log_seqs, positions):
        seqs = self._get_embedding(log_seqs)
        if hasattr(self.item_emb, 'embedding_dim'):
            seqs *= self.item_emb.embedding_dim ** 0.5
        else:
            seqs *= 64 ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)

        log_feats = self.backbone(seqs, log_seqs)

        return log_feats

    def forward(self, seq, pos, neg, positions):
        log_feats = self.log2feats(seq, positions)
        log_feats = log_feats[:, -1, :].unsqueeze(1)

        pos_embs = self._get_embedding(pos.unsqueeze(1))
        neg_embs = self._get_embedding(neg)

        pos_logits = torch.mul(log_feats, pos_embs).sum(dim=-1)
        neg_logits = torch.mul(log_feats, neg_embs).sum(dim=-1)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.dev), torch.zeros(neg_logits.shape, device=self.dev)
        indices = (pos != 0)
        pos_loss, neg_loss = self.loss_func(pos_logits[indices], pos_labels[indices]), self.loss_func(neg_logits[indices], neg_labels[indices])
        loss = pos_loss + neg_loss

        return loss

    def predict(self, seq, item_indices, positions, **kwargs):
        log_feats = self.log2feats(seq, positions)
        final_feat = log_feats[:, -1, :]
        item_embs = self._get_embedding(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits

    def get_user_emb(self, seq, positions, **kwargs):
        log_feats = self.log2feats(seq, positions)
        final_feat = log_feats[:, -1, :]
        
        return final_feat
    
    def get_parameter_groups(self, codebook_lr=None, projection_lr=None, decoder_lr=None, backbone_lr=None):
        param_groups = []
        
        if self.use_semantic_id and hasattr(self, 'use_rqvae_embedding') and self.use_rqvae_embedding:
            if hasattr(self.item_emb, 'get_codebook_parameters'):
                codebook_params = self.item_emb.get_codebook_parameters()
                if codebook_params and codebook_lr is not None:
                    param_groups.append({
                        'params': codebook_params,
                        'lr': codebook_lr,
                        'name': 'codebook'
                    })
            
            if hasattr(self.item_emb, 'get_projection_parameters'):
                projection_params = self.item_emb.get_projection_parameters()
                if projection_params and projection_lr is not None:
                    param_groups.append({
                        'params': projection_params,
                        'lr': projection_lr,
                        'name': 'projection'
                    })
            
            if hasattr(self.item_emb, 'get_decoder_parameters'):
                decoder_params = self.item_emb.get_decoder_parameters()
                if decoder_params and decoder_lr is not None:
                    param_groups.append({
                        'params': decoder_params,
                        'lr': decoder_lr,
                        'name': 'decoder'
                    })
        
        other_params = []
        for name, param in self.named_parameters():
            if self.use_semantic_id and hasattr(self, 'use_rqvae_embedding') and self.use_rqvae_embedding:
                if 'item_emb.codebook_embeddings' in name:
                    continue
                if any(proj_name in name for proj_name in [
                    'item_emb.projection', 'item_emb.projections', 'item_emb.final_projection',
                    'item_emb.decoder_input_projection', 'item_emb.decoder_input_compression', 
                    'item_emb.decoder_output_projection'
                ]):
                    continue
                if 'item_emb.decoder.mlp' in name:
                    continue
            
            other_params.append(param)
        
        if other_params:
            lr = backbone_lr if backbone_lr is not None else 1e-3
            param_groups.append({
                'params': other_params,
                'lr': lr,
                'name': 'backbone'
            })
        
        return param_groups
    
    def save_codebooks(self, save_path: str):
        if self.use_semantic_id and hasattr(self, 'use_rqvae_embedding') and self.use_rqvae_embedding:
            if hasattr(self.item_emb, 'save_updated_codebooks'):
                self.item_emb.save_updated_codebooks(save_path)
                print(f"[SemanticSASRec] 训练后的码本权重已保存到: {save_path}")
            else:
                print("[SemanticSASRec] 当前embedding层不支持码本保存")
        else:
            print("[SemanticSASRec] 未使用RQVAE embedding，无需保存码本")
    
    def freeze_codebooks(self):
        if self.use_semantic_id and hasattr(self, 'use_rqvae_embedding') and self.use_rqvae_embedding:
            if hasattr(self.item_emb, 'freeze_codebooks'):
                self.item_emb.freeze_codebooks()
    
    def unfreeze_codebooks(self):
        if self.use_semantic_id and hasattr(self, 'use_rqvae_embedding') and self.use_rqvae_embedding:
            if hasattr(self.item_emb, 'unfreeze_codebooks'):
                self.item_emb.unfreeze_codebooks()
    
    def save_item_embeddings(self, save_path: str):
        print(f"[SemanticSASRec] 开始保存item embeddings到: {save_path}")
        
        if not self.use_semantic_id:
            print("[SemanticSASRec] 当前模型未使用semantic ID，无法保存item embeddings")
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        device = next(self.parameters()).device
        all_item_ids = torch.arange(1, self.item_num + 1, device=device)
        
        batch_size = 1000
        all_embeddings = []
        
        self.eval()
        with torch.no_grad():
            for i in range(0, len(all_item_ids), batch_size):
                batch_ids = all_item_ids[i:i+batch_size]
                batch_embs = self._get_embedding(batch_ids)
                all_embeddings.append(batch_embs.cpu().numpy())
        
        item_embeddings = np.concatenate(all_embeddings, axis=0)
        
        save_data = {
            'embeddings': item_embeddings,
            'item_num': self.item_num,
            'embedding_dim': item_embeddings.shape[1],
            'model_type': 'SemanticSASRec',
            'use_semantic_id': self.use_semantic_id
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"[SemanticSASRec] 成功保存 {item_embeddings.shape[0]} 个item embeddings")
        print(f"[SemanticSASRec] Embedding维度: {item_embeddings.shape[1]}")
        print(f"[SemanticSASRec] 保存路径: {save_path}")
        
        return save_path


class SemanticSASRec_seq(SemanticSASRec):

    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)

    def forward(self, seq, pos, neg, positions, **kwargs):
        log_feats = self.log2feats(seq, positions)
        pos_embs = self._get_embedding(pos)
        neg_embs = self._get_embedding(neg)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.dev), torch.zeros(neg_logits.shape, device=self.dev)
        indices = (pos != 0)
        pos_loss, neg_loss = self.loss_func(pos_logits[indices], pos_labels[indices]), self.loss_func(neg_logits[indices], neg_labels[indices])
        loss = pos_loss + neg_loss

        return loss
