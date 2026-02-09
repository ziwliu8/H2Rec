import pickle
import numpy as np
import torch
import torch.nn as nn
from models.BaseModel import BaseSeqModel
from models.SASRec import SASRecBackbone
from models.utils import PointWiseFeedForward


class LLM_SASRec(BaseSeqModel):
    
    def __init__(self, user_num, item_num, device, args):
        super(LLM_SASRec, self).__init__(user_num, item_num, device, args)
        
        try:
            llm_emb_path = f"./data/{args.dataset}/handled/itm_emb_np.pkl"
            with open(llm_emb_path, "rb") as f:
                llm_emb_all = pickle.load(f)
            
            llm_item_emb = np.concatenate([
                np.zeros((1, llm_emb_all.shape[1])),
                llm_emb_all
            ])
            
            if hasattr(args, 'use_llm_emb') and args.use_llm_emb:
                self.item_emb_llm = nn.Embedding.from_pretrained(
                    torch.Tensor(llm_item_emb), 
                    padding_idx=0
                )
                if hasattr(args, 'freeze_llm_emb') and args.freeze_llm_emb:
                    self.item_emb_llm.weight.requires_grad = False
                else:
                    self.item_emb_llm.weight.requires_grad = True
            else:
                self.item_emb_llm = nn.Embedding(
                    self.item_num + 1, 
                    args.hidden_size, 
                    padding_idx=0
                )
            
            if hasattr(args, 'use_llm_emb') and args.use_llm_emb:
                self.adapter = nn.Sequential(
                    nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
                    nn.ReLU(),
                    nn.Dropout(args.dropout_rate),
                    nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
                )
            else:
                self.adapter = nn.Identity()
            
        except FileNotFoundError:
            print(f"Warning: LLM embedding file not found at ./data/{args.dataset}/handled/item_emb.pkl")
            print("Using random initialized embeddings instead.")
            self.item_emb_llm = nn.Embedding(
                self.item_num + 1, 
                args.hidden_size, 
                padding_idx=0
            )
            self.adapter = nn.Identity()
        
        self.pos_emb = nn.Embedding(args.max_len + 100, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        
        self.backbone = SASRecBackbone(device, args)
        
        self.loss_func = nn.BCEWithLogitsLoss()
        
        if hasattr(args, 'use_llm_emb') and args.use_llm_emb:
            self.filter_init_modules.append("item_emb_llm")
        
        self._init_weights()
    
    def _get_embedding(self, log_seqs):
        item_seq_emb = self.item_emb_llm(log_seqs)
        item_seq_emb = self.adapter(item_seq_emb)
        return item_seq_emb
    
    def log2feats(self, log_seqs, positions):
        seqs = self._get_embedding(log_seqs)
        seqs *= self.item_emb_llm.embedding_dim ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)
        
        log_feats = self.backbone(seqs, log_seqs)
        return log_feats
    
    def forward(self, seq, pos, neg, positions, **kwargs):
        log_feats = self.log2feats(seq, positions)
        log_feats = log_feats[:, -1, :].unsqueeze(1)
        
        pos_embs = self._get_embedding(pos.unsqueeze(1))
        neg_embs = self._get_embedding(neg)
        
        pos_logits = torch.mul(log_feats, pos_embs).sum(dim=-1)
        neg_logits = torch.mul(log_feats, neg_embs).sum(dim=-1)
        
        pos_labels = torch.ones(pos_logits.shape, device=self.dev)
        neg_labels = torch.zeros(neg_logits.shape, device=self.dev)
        
        indices = (pos != 0)
        pos_loss = self.loss_func(pos_logits[indices], pos_labels[indices])
        neg_loss = self.loss_func(neg_logits[indices], neg_labels[indices])
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
    
    def get_log_feats(self, seq, positions, **kwargs):
        log_feats = self.log2feats(seq, positions)
        final_feat = log_feats[:, -1, :]
        
        return final_feat


class LLM_SASRec_seq(LLM_SASRec):
    
    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
    
    def forward(self, seq, pos, neg, positions, **kwargs):
        log_feats = self.log2feats(seq, positions)
        pos_embs = self._get_embedding(pos)
        neg_embs = self._get_embedding(neg)
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        
        pos_labels = torch.ones(pos_logits.shape, device=self.dev)
        neg_labels = torch.zeros(neg_logits.shape, device=self.dev)
        
        indices = (pos != 0)
        pos_loss = self.loss_func(pos_logits[indices], pos_labels[indices])
        neg_loss = self.loss_func(neg_logits[indices], neg_labels[indices])
        loss = pos_loss + neg_loss
        
        return loss
