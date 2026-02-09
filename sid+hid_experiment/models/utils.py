import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
    


class Contrastive_Loss2(nn.Module):

    def __init__(self, tau=1) -> None:
        super().__init__()

        self.temperature = tau


    def forward(self, X, Y):
        
        logits = (X @ Y.T) / self.temperature
        X_similarity = Y @ Y.T
        Y_similarity = X @ X.T
        targets = F.softmax(
            (X_similarity + Y_similarity) / 2 * self.temperature, dim=-1
        )
        X_loss = self.cross_entropy(logits, targets, reduction='none')
        Y_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (Y_loss + X_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    

    def cross_entropy(self, preds, targets, reduction='none'):

        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
    


class InfoNCE_Loss(nn.Module):
    def __init__(self, temperature=0.07, use_hard_negatives=True, debias=True):
        super().__init__()
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        self.debias = debias  # 去偏处理，减少假负样本影响
    
    def forward(self, anchor, positive, negatives=None, positive_mask=None, 
                positive_weights=None, hard_negatives=None):

        batch_size = anchor.size(0)
        device = anchor.device
        
        # 归一化
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        
        # 计算锚点与所有正样本的相似度 [N, N]
        sim_matrix = torch.matmul(anchor, positive.t()) / self.temperature
        
        # 构建正样本mask（默认对角线）
        if positive_mask is None:
            positive_mask = torch.eye(batch_size, device=device)
        
        # 构建正样本权重（默认等权）
        if positive_weights is None:
            positive_weights = positive_mask
        else:
            # 确保权重只作用在正样本上
            positive_weights = positive_weights * positive_mask
        
        # 计算损失
        exp_sim = torch.exp(sim_matrix)
        
        # 分子：加权正样本的exp相似度和
        # 使用权重矩阵对正样本进行加权
        weighted_pos_sim = (exp_sim * positive_weights).sum(dim=1)
        
        # 分母：所有样本的exp相似度和（in-batch negatives）
        all_sim = exp_sim.sum(dim=1)
        
        # 添加硬负样本到分母
        if hard_negatives is not None and self.use_hard_negatives:
            hard_negatives = F.normalize(hard_negatives, dim=-1)
            # [N, K]
            hard_neg_sim = torch.matmul(anchor.unsqueeze(1), hard_negatives.transpose(1, 2)).squeeze(1)
            hard_neg_sim = hard_neg_sim / self.temperature
            hard_neg_exp = torch.exp(hard_neg_sim).sum(dim=1)
            all_sim = all_sim + hard_neg_exp
        
        # 去偏处理：减少假负样本的影响
        if self.debias:
            # 估计假负样本数量并调整分母
            # 简化版本：假设batch中有10%是潜在的假负样本
            neg_count = batch_size - positive_mask.sum(dim=1)
            debiasing_term = neg_count * 0.1 * torch.exp(torch.tensor(0.0))
            all_sim = all_sim - debiasing_term
        
        # InfoNCE loss: -log(加权正样本相似度 / 所有相似度)
        loss = -torch.log(weighted_pos_sim / (all_sim + 1e-8) + 1e-8)
        
        return loss.mean()


class CalculateAttention(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, Q, K, V, mask):

        attention = torch.matmul(Q,torch.transpose(K, -1, -2))
        # use mask
        attention = attention.masked_fill_(mask, -1e9)
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention,V)
        return attention


#y的信息通过注意力机制注入到x中，x作为接收信息的主体，y作为提供信息的源头
class Multi_CrossAttention(nn.Module):
    """
    forward时，第一个参数用于计算query，第二个参数用于计算key和value
    """
    def __init__(self,hidden_size,all_head_size,head_num):
        super().__init__()
        self.hidden_size    = hidden_size       # 输入维度
        self.all_head_size  = all_head_size     # 输出维度
        self.num_heads      = head_num          # 注意头的数量
        self.h_size         = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)


    def print(self):
        print(self.hidden_size,self.all_head_size)
        print(self.linear_k,self.linear_q,self.linear_v)
    

    def forward(self,x,y,log_seqs):
        """
        cross-attention: x,y是两个模型的隐藏层，将x作为q的输入，y作为k和v的输入
        """

        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # attention_mask = attention_mask.eq(0)
        attention_mask = (log_seqs == 0).unsqueeze(1).repeat(1, log_seqs.size(1), 1).unsqueeze(1)

        attention = CalculateAttention()(q_s,k_s,v_s,attention_mask)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        
        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output



class Attention(nn.Module):

    def __init__(self, hidden_size, method="dot"):

        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method == "dot":
            pass
        elif self.method == "general":
            self.Wa = nn.Linear(hidden_size, hidden_size,bias=False)


    def forward(self, query, key):
        """
        query: [bs, hidden_size]
        key: [bs, seq_len, hidden_size]
        weight: [bs, seq_len, 1]
        """

        if self.method == "dot":
            return self.dot_score(query, key)
        elif self.method == "general":
            return self.general_score(query, key)


    def dot_score(self, query, key):
        
        query = query.unsqueeze(2)  #[bs, hidden_size, 1]
        attn_energies = torch.bmm(key, query) # (bs, seq_len, hidden_size) * (bs, hidden_size, 1) --> (bs, seq_len, 1)
        attn_energies = attn_energies.squeeze(-1) # (bs, seq_len)

        return F.softmax(attn_energies, dim=-1).unsqueeze(-1)  # [batch_size, seq_len, 1]
    

    def general_score(self, query, key):

        query = self.Wa(query).unsqueeze(2) # (bs, hidden_size, 1)
        attn_energies = torch.bmm(key, query).squeeze(-1) 
        
        return F.softmax(attn_energies,dim=-1).unsqueeze(-1)


# ============================================================================
# 对比学习增强模块
# ============================================================================

class SequenceAugmenter(nn.Module):
    """
    CL4SRec风格的序列增强
    支持：crop, mask, reorder增强策略
    """
    def __init__(self, aug_types=['crop', 'mask', 'reorder'], 
                 crop_ratio=0.2, mask_ratio=0.2, reorder_ratio=0.2):
        super().__init__()
        self.aug_types = aug_types
        self.crop_ratio = crop_ratio
        self.mask_ratio = mask_ratio
        self.reorder_ratio = reorder_ratio
    
    def forward(self, seq, mask_token=0):
        """
        Args:
            seq: [batch_size, seq_len] 原始序列
            mask_token: mask token的ID
        Returns:
            aug_seq1, aug_seq2: 两个增强视图
        """
        aug_seq1 = self.random_augment(seq, mask_token)
        aug_seq2 = self.random_augment(seq, mask_token)
        return aug_seq1, aug_seq2
    
    def random_augment(self, seq, mask_token):
        """随机选择一种增强策略"""
        import random
        aug_type = random.choice(self.aug_types)
        
        if aug_type == 'crop':
            return self.crop(seq)
        elif aug_type == 'mask':
            return self.mask(seq, mask_token)
        elif aug_type == 'reorder':
            return self.reorder(seq)
        else:
            return seq
    
    def crop(self, seq):
        """裁剪：随机保留连续的一段"""
        batch_size, seq_len = seq.size()
        crop_len = max(1, int(seq_len * (1 - self.crop_ratio)))
        
        aug_seq = seq.clone()
        for i in range(batch_size):
            # 找到有效长度
            valid_len = (seq[i] != 0).sum().item()
            if valid_len <= 1:
                continue
            
            # 随机选择起始位置
            start = torch.randint(0, max(1, valid_len - crop_len + 1), (1,)).item()
            end = start + crop_len
            
            # 裁剪并左对齐
            aug_seq[i, :crop_len] = seq[i, start:end]
            aug_seq[i, crop_len:] = 0
        
        return aug_seq
    
    def mask(self, seq, mask_token):
        """掩码：随机mask一些位置"""
        aug_seq = seq.clone()
        batch_size, seq_len = seq.size()
        
        for i in range(batch_size):
            valid_len = (seq[i] != 0).sum().item()
            if valid_len <= 1:
                continue
            
            # 随机选择mask位置
            n_mask = max(1, int(valid_len * self.mask_ratio))
            mask_indices = torch.randperm(valid_len)[:n_mask]
            aug_seq[i, mask_indices] = mask_token
        
        return aug_seq
    
    def reorder(self, seq):
        """重排序：随机打乱一小段"""
        aug_seq = seq.clone()
        batch_size, seq_len = seq.size()
        
        for i in range(batch_size):
            valid_len = (seq[i] != 0).sum().item()
            if valid_len <= 2:
                continue
            
            # 随机选择重排序段
            reorder_len = max(2, int(valid_len * self.reorder_ratio))
            start = torch.randint(0, max(1, valid_len - reorder_len + 1), (1,)).item()
            end = start + reorder_len
            
            # 打乱这一段
            segment = aug_seq[i, start:end].clone()
            perm = torch.randperm(reorder_len)
            aug_seq[i, start:end] = segment[perm]
        
        return aug_seq


class HardNegativeSampler(nn.Module):
    """
    硬负样本采样器
    策略：采样语义相近但不在用户交互中的物品
    """
    def __init__(self, item_embeddings, n_hard_negatives=5, temperature=0.5):
        super().__init__()
        self.item_embeddings = item_embeddings  # 预计算的物品embedding
        self.n_hard_negatives = n_hard_negatives
        self.temperature = temperature
    
    def forward(self, anchor_items, user_histories, top_k=100):
        """
        Args:
            anchor_items: [batch_size] 锚点物品ID
            user_histories: [batch_size, seq_len] 用户历史交互
            top_k: 从最相似的top_k中采样
        Returns:
            hard_negatives: [batch_size, n_hard_negatives] 硬负样本ID
        """
        batch_size = anchor_items.size(0)
        device = anchor_items.device
        
        # 获取锚点物品的embedding
        anchor_emb = self.item_embeddings(anchor_items)  # [batch_size, dim]
        
        # 计算与所有物品的相似度
        all_item_emb = self.item_embeddings.weight  # [n_items, dim]
        similarities = torch.matmul(anchor_emb, all_item_emb.t())  # [batch_size, n_items]
        
        # 对每个样本，过滤掉用户历史中的物品
        hard_neg_items = []
        for i in range(batch_size):
            # 用户历史物品
            history = set(user_histories[i].cpu().numpy().tolist())
            history.discard(0)  # 移除padding
            history.add(anchor_items[i].item())  # 移除anchor自己
            
            # 找到top_k相似但不在历史中的物品
            sim_scores = similarities[i].clone()
            for item_id in history:
                sim_scores[item_id] = -float('inf')
            
            # 温度采样：相似度越高，采样概率越大
            probs = F.softmax(sim_scores / self.temperature, dim=0)
            
            # 采样n_hard_negatives个
            sampled = torch.multinomial(probs, self.n_hard_negatives, replacement=False)
            hard_neg_items.append(sampled)
        
        hard_negatives = torch.stack(hard_neg_items, dim=0)  # [batch_size, n_hard_negatives]
        
        return hard_negatives


class DynamicPositiveFilter(nn.Module):
    """
    动态正样本筛选器
    基于序列表示的相似度，过滤弱相关的窗口正样本
    """
    def __init__(self, top_k_ratio=0.5, min_similarity=0.1):
        super().__init__()
        self.top_k_ratio = top_k_ratio  # 保留前k%的正样本
        self.min_similarity = min_similarity  # 最小相似度阈值
    
    def forward(self, seq_repr, positive_items_emb, positive_mask):
        """
        Args:
            seq_repr: [batch_size, dim] 序列表示
            positive_items_emb: [batch_size, n_candidates, dim] 候选正样本embedding
            positive_mask: [batch_size, n_candidates] 初始正样本mask
        Returns:
            filtered_mask: [batch_size, n_candidates] 过滤后的正样本mask
            filtered_weights: [batch_size, n_candidates] 正样本权重
        """
        batch_size = seq_repr.size(0)
        n_candidates = positive_items_emb.size(1)
        device = seq_repr.device
        
        # 计算序列表示与每个正样本的相似度
        seq_repr = F.normalize(seq_repr, dim=-1)
        positive_items_emb = F.normalize(positive_items_emb, dim=-1)
        
        # [batch_size, n_candidates]
        similarities = torch.bmm(
            seq_repr.unsqueeze(1), 
            positive_items_emb.transpose(1, 2)
        ).squeeze(1)
        
        # 只考虑初始正样本
        similarities = similarities * positive_mask
        similarities[positive_mask == 0] = -float('inf')
        
        # 动态阈值：保留top_k_ratio的正样本
        filtered_mask = torch.zeros_like(positive_mask)
        filtered_weights = torch.zeros_like(positive_mask, dtype=torch.float)
        
        for i in range(batch_size):
            n_positives = positive_mask[i].sum().item()
            if n_positives == 0:
                continue
            
            k = max(1, int(n_positives * self.top_k_ratio))
            
            # 获取top-k相似的正样本
            valid_sims = similarities[i][positive_mask[i] == 1]
            threshold = torch.topk(valid_sims, k).values[-1].item()
            threshold = max(threshold, self.min_similarity)
            
            # 过滤并赋权
            keep_indices = (similarities[i] >= threshold) & (positive_mask[i] == 1)
            filtered_mask[i] = keep_indices.float()
            
            # 权重归一化到[0.3, 1.0]
            if keep_indices.sum() > 0:
                weights = similarities[i][keep_indices]
                weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
                weights = 0.3 + 0.7 * weights  # 映射到[0.3, 1.0]
                filtered_weights[i][keep_indices] = weights
        
        return filtered_mask, filtered_weights


class ViewReconstructionHead(nn.Module):
    """
    序列级视角重构头部（SMVM - Sequence-level Masked View Modeling）
    
    功能：从其他视角和Hash特征重构被遮蔽的视角表征
    支持两种监督方式：
    1. 回归：预测被遮蔽视角的表征向量（Cosine/L2损失）
    2. 分类：预测被遮蔽视角的码字索引（交叉熵损失）
    """
    
    def __init__(self, input_dim, output_dim, n_codebooks=4, codebook_size=128, 
                 recon_method='cosine', dropout=0.1):
        """
        Args:
            input_dim: 输入特征维度（通常是hidden_size）
            output_dim: 输出特征维度（重构的视角表征维度）
            n_codebooks: 码本数量（用于分类模式）
            codebook_size: 每个码本的大小（用于分类模式）
            recon_method: 重构方法，'cosine'/'l2'（回归）或 'code_cls'（分类）
            dropout: dropout率
        """
        super().__init__()
        self.recon_method = recon_method
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        
        if recon_method in ['cosine', 'l2']:
            # 回归模式：轻量投影网络
            self.projection = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim, output_dim)
            )
        elif recon_method == 'code_cls':
            # 分类模式：为每个码本预测码字索引
            self.classifiers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(input_dim // 2, codebook_size)
                )
                for _ in range(n_codebooks)
            ])
        else:
            raise ValueError(f"Unknown recon_method: {recon_method}")
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch_size, seq_len, input_dim] 序列隐状态
        Returns:
            如果是回归：[batch_size, seq_len, output_dim] 重构的表征
            如果是分类：list of [batch_size, seq_len, codebook_size] logits
        """
        if self.recon_method in ['cosine', 'l2']:
            return self.projection(hidden_states)
        elif self.recon_method == 'code_cls':
            # 返回每个码本的分类logits
            return [classifier(hidden_states) for classifier in self.classifiers]
    
    def compute_loss(self, predictions, targets, mask=None):
        """
        计算重构损失
        
        Args:
            predictions: 模型预测
                - 回归模式: [batch_size, seq_len, dim]
                - 分类模式: list of [batch_size, seq_len, codebook_size]
            targets: 真实目标
                - 回归模式: [batch_size, seq_len, dim] 原始视角表征
                - 分类模式: [batch_size, seq_len, n_codebooks] 码字索引
            mask: [batch_size, seq_len] 有效位置mask（0表示padding）
        Returns:
            loss: 标量损失值
        """
        if self.recon_method == 'cosine':
            # Cosine相似度损失（1 - cosine_sim）
            predictions = F.normalize(predictions, dim=-1)
            targets = F.normalize(targets, dim=-1)
            loss = 1 - (predictions * targets).sum(dim=-1)  # [batch_size, seq_len]
            
        elif self.recon_method == 'l2':
            # L2距离损失
            loss = F.mse_loss(predictions, targets, reduction='none').mean(dim=-1)
            
        elif self.recon_method == 'code_cls':
            # 交叉熵分类损失（对每个码本）
            batch_size, seq_len = targets.shape[:2]
            total_loss = 0
            
            for i, pred_logits in enumerate(predictions):
                # pred_logits: [batch_size, seq_len, codebook_size]
                # targets[:, :, i]: [batch_size, seq_len] 第i个码本的目标索引
                target_codes = targets[:, :, i].long()
                
                # 展平计算交叉熵
                pred_flat = pred_logits.view(-1, self.codebook_size)
                target_flat = target_codes.view(-1)
                
                loss_i = F.cross_entropy(pred_flat, target_flat, reduction='none')
                loss_i = loss_i.view(batch_size, seq_len)
                total_loss += loss_i
            
            loss = total_loss / self.n_codebooks  # 平均到每个码本
        
        else:
            raise ValueError(f"Unknown recon_method: {self.recon_method}")
        
        # 应用mask（只计算有效位置的损失）
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss


