import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import json
import numpy as np
import sys
from typing import List, Dict, Optional, Union


class RQVAECodebookEmbedding(nn.Module):
    
    def __init__(
        self, 
        codebook_file: str,
        semantic_code_file: str,
        embedding_dim: int,
        padding_idx: int = 0,
        fusion_method: str = 'sum',
        learnable: bool = True,
        device: str = 'cuda',
        use_decoder: bool = False,
        model_file: str = None,
        freeze_decoder: bool = True
    ):

        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.fusion_method = fusion_method
        self.learnable = learnable
        self.device = device
        self.use_decoder = use_decoder
        self.model_file = model_file
        self.freeze_decoder = freeze_decoder
        
        self._load_codebooks_and_codes(codebook_file, semantic_code_file)
        self._create_codebook_embeddings()
        
        if self.use_decoder:
            self._load_decoder()
        
        self._create_projection_layer()
    
    def _load_codebooks_and_codes(self, codebook_file: str, semantic_code_file: str):
        with open(codebook_file, 'rb') as f:
            codebook_dict = pickle.load(f)
        
        self.codebooks = []
        self.codebook_names = []
        
        if 'cluster_layer' in codebook_dict:
            self.codebooks.append(codebook_dict['cluster_layer'])
            self.codebook_names.append('cluster_layer')
        
        rq_layers = [k for k in codebook_dict.keys() if k.startswith('rq_layer_')]
        rq_layers.sort(key=lambda x: int(x.split('_')[-1]))
        for layer_name in rq_layers:
            self.codebooks.append(codebook_dict[layer_name])
            self.codebook_names.append(layer_name)
        
        if 'euclidean_layer' in codebook_dict:
            self.codebooks.append(codebook_dict['euclidean_layer'])
            self.codebook_names.append('euclidean_layer')
        
        self.n_codebooks = len(self.codebooks)
        self.codebook_dims = [cb.shape[1] for cb in self.codebooks]
        self.codebook_sizes = [cb.shape[0] for cb in self.codebooks]
        
        with open(semantic_code_file, 'r') as f:
            semantic_codes_list = json.load(f)
        
        self.semantic_codes = semantic_codes_list
        self.n_items = len(semantic_codes_list)
        
        self._build_lookup_table()
    
    def _build_lookup_table(self):
        lookup_table = torch.zeros(self.n_items, self.n_codebooks, dtype=torch.long)
        
        for item_id, semantic_code in enumerate(self.semantic_codes):
            if semantic_code:
                if len(semantic_code) == self.n_codebooks:
                    lookup_table[item_id] = torch.tensor(semantic_code, dtype=torch.long)
        
        self.register_buffer('semantic_lookup_table', lookup_table)
    
    def _create_codebook_embeddings(self):
        self.codebook_embeddings = nn.ModuleList()
        
        for i, (codebook, name) in enumerate(zip(self.codebooks, self.codebook_names)):
            codebook_tensor = torch.from_numpy(codebook).float()
            
            if self.learnable:
                embedding_layer = nn.Embedding(
                    num_embeddings=codebook_tensor.shape[0],
                    embedding_dim=codebook_tensor.shape[1],
                    padding_idx=None
                )
                embedding_layer.weight.data.copy_(codebook_tensor)
            else:
                embedding_layer = nn.Embedding.from_pretrained(
                    codebook_tensor, 
                    freeze=True,
                    padding_idx=None
                )
            
            self.codebook_embeddings.append(embedding_layer)
    
    def _load_decoder(self):
        if not self.model_file:
            raise ValueError("使用decoder时必须提供model_file路径")
        
        checkpoint = torch.load(self.model_file, map_location='cpu', weights_only=False)
        model_state = checkpoint['model_state_dict']
        config = checkpoint['config']
        
        embed_dim = config.get('embed_dim', self.codebook_dims[0])
        input_dim = config.get('input_dim', 1536)
        hidden_dim = config.get('hidden_dim', 512)
        
        sys.path.append('/input0/sid+hid_experiment/Unified_Semantic_ID-9E94')
        from rq_modules.decoder import Decode_MLP
        
        self.decoder = Decode_MLP(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=input_dim
        )
        
        decoder_state = {}
        for key, value in model_state.items():
            if key.startswith('decoder.'):
                new_key = key[8:]
                decoder_state[new_key] = value
        
        self.decoder.load_state_dict(decoder_state)
        
        if self.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
        
        self.decoder_input_dim = embed_dim
        self.decoder_output_dim = input_dim
    
    def _create_projection_layer(self):
        if self.fusion_method == 'concat':
            total_dim = sum(self.codebook_dims)
            if total_dim != self.embedding_dim:
                self.projection = nn.Linear(total_dim, self.embedding_dim)
            else:
                self.projection = None
        elif self.fusion_method in ['sum', 'mean']:
            if len(set(self.codebook_dims)) > 1:
                target_dim = max(self.codebook_dims)
                self.projections = nn.ModuleList()
                for dim in self.codebook_dims:
                    if dim != target_dim:
                        self.projections.append(nn.Linear(dim, target_dim))
                    else:
                        self.projections.append(nn.Identity())
            else:
                target_dim = self.codebook_dims[0]
                self.projections = None
            
            if self.use_decoder:
                self.final_projection = None
            else:
                if target_dim != self.embedding_dim:
                    self.final_projection = nn.Sequential(
                        nn.Linear(target_dim, target_dim // 2),
                        nn.Linear(target_dim // 2, self.embedding_dim)
                    )
                else:
                    self.final_projection = None
    
    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        original_shape = item_ids.shape
        item_ids_flat = item_ids.view(-1)
        batch_size = item_ids_flat.size(0)
        
        item_ids_safe = torch.clamp(item_ids_flat, 0, self.n_items - 1)
        semantic_codes = self.semantic_lookup_table[item_ids_safe]
        
        codebook_embeddings = []
        for codebook_idx in range(self.n_codebooks):
            codes = semantic_codes[:, codebook_idx]
            codes = torch.clamp(codes, 0, self.codebook_sizes[codebook_idx] - 1)
            emb = self.codebook_embeddings[codebook_idx](codes)
            codebook_embeddings.append(emb)
        
        if self.fusion_method == 'concat':
            fused_emb = torch.cat(codebook_embeddings, dim=-1)
            if self.projection is not None:
                fused_emb = self.projection(fused_emb)
                
        elif self.fusion_method in ['sum', 'mean']:
            if self.projections is not None:
                aligned_embeddings = []
                for emb, proj in zip(codebook_embeddings, self.projections):
                    aligned_embeddings.append(proj(emb))
                codebook_embeddings = aligned_embeddings
            
            fused_emb = torch.stack(codebook_embeddings, dim=0).sum(dim=0)
            if self.fusion_method == 'mean':
                fused_emb = fused_emb / self.n_codebooks
            
            if self.final_projection is not None:
                fused_emb = self.final_projection(fused_emb)
        
        else:
            raise ValueError(f"不支持的融合方法: {self.fusion_method}")
        
        if self.use_decoder:
            if fused_emb.shape[-1] == self.decoder_input_dim:
                decoder_output = self.decoder(fused_emb)
            else:
                if fused_emb.shape[-1] < self.decoder_input_dim:
                    if not hasattr(self, 'decoder_input_projection'):
                        self.decoder_input_projection = nn.Linear(
                            fused_emb.shape[-1], 
                            self.decoder_input_dim
                        ).to(self.device)
                    
                    fused_emb = self.decoder_input_projection(fused_emb)
                    decoder_output = self.decoder(fused_emb)
                else:
                    if not hasattr(self, 'decoder_input_compression'):
                        self.decoder_input_compression = nn.Linear(
                            fused_emb.shape[-1], 
                            self.decoder_input_dim
                        ).to(self.device)
                    
                    compressed_emb = self.decoder_input_compression(fused_emb)
                    decoder_output = self.decoder(compressed_emb)
            
            if decoder_output.shape[-1] != self.embedding_dim:
                if not hasattr(self, 'decoder_output_projection'):
                    self.decoder_output_projection = nn.Linear(
                        decoder_output.shape[-1], 
                        self.embedding_dim
                    ).to(self.device)
                
                decoder_output = self.decoder_output_projection(decoder_output)
            
            fused_emb = decoder_output
        
        padding_mask = (item_ids_flat == self.padding_idx).unsqueeze(-1)
        fused_emb = fused_emb * (~padding_mask).float()
        
        if len(original_shape) == 1:
            return fused_emb
        else:
            return fused_emb.view(original_shape[0], original_shape[1], -1)
    
    def get_per_view_embeddings(self, item_ids: torch.Tensor, project_to_dim: Optional[int] = None) -> List[torch.Tensor]:
        target_out_dim = project_to_dim if project_to_dim is not None else self.embedding_dim
        original_shape = item_ids.shape
        item_ids_flat = item_ids.view(-1)
        
        item_ids_safe = torch.clamp(item_ids_flat, 0, self.n_items - 1)
        semantic_codes = self.semantic_lookup_table[item_ids_safe]
        
        per_view_emb_list = []
        for codebook_idx in range(self.n_codebooks):
            codes = semantic_codes[:, codebook_idx]
            codes = torch.clamp(codes, 0, self.codebook_sizes[codebook_idx] - 1)
            emb = self.codebook_embeddings[codebook_idx](codes)
            per_view_emb_list.append(emb)
        
        if self.projections is not None:
            aligned_list = []
            for emb, proj in zip(per_view_emb_list, self.projections):
                aligned_list.append(proj(emb))
            per_view_emb_list = aligned_list
            unified_dim = per_view_emb_list[0].shape[-1]
        else:
            unified_dim = per_view_emb_list[0].shape[-1]
        
        if unified_dim != target_out_dim:
            if not hasattr(self, 'per_view_projection') or self.per_view_projection is None:
                self.per_view_projection = nn.Linear(unified_dim, target_out_dim).to(self.device)
            per_view_emb_list = [self.per_view_projection(emb) for emb in per_view_emb_list]
        
        padding_mask = (item_ids_flat == self.padding_idx).unsqueeze(-1)
        per_view_emb_list = [emb * (~padding_mask).float() for emb in per_view_emb_list]
        
        if len(original_shape) == 1:
            return per_view_emb_list
        else:
            return [emb.view(original_shape[0], original_shape[1], -1) for emb in per_view_emb_list]
    
    def get_codebook_parameters(self) -> List[nn.Parameter]:
        params = []
        for emb_layer in self.codebook_embeddings:
            params.extend(emb_layer.parameters())
        return params
    
    def get_projection_parameters(self) -> List[nn.Parameter]:
        params = []
        if hasattr(self, 'projection') and self.projection is not None:
            params.extend(self.projection.parameters())
        if hasattr(self, 'projections') and self.projections is not None:
            for proj in self.projections:
                if not isinstance(proj, nn.Identity):
                    params.extend(proj.parameters())
        if hasattr(self, 'final_projection') and self.final_projection is not None:
            params.extend(self.final_projection.parameters())
        
        if hasattr(self, 'decoder_input_projection') and self.decoder_input_projection is not None:
            params.extend(self.decoder_input_projection.parameters())
        if hasattr(self, 'decoder_input_compression') and self.decoder_input_compression is not None:
            params.extend(self.decoder_input_compression.parameters())
        if hasattr(self, 'decoder_output_projection') and self.decoder_output_projection is not None:
            params.extend(self.decoder_output_projection.parameters())
        if hasattr(self, 'per_view_projection') and self.per_view_projection is not None:
            params.extend(self.per_view_projection.parameters())
        
        return params
    
    def get_decoder_parameters(self) -> List[nn.Parameter]:
        params = []
        if hasattr(self, 'decoder') and self.decoder is not None:
            params.extend(self.decoder.parameters())
        return params
    
    def freeze_codebooks(self):
        for emb_layer in self.codebook_embeddings:
            for param in emb_layer.parameters():
                param.requires_grad = False
    
    def unfreeze_codebooks(self):
        for emb_layer in self.codebook_embeddings:
            for param in emb_layer.parameters():
                param.requires_grad = True
    
    def freeze_decoder(self):
        if hasattr(self, 'decoder') and self.decoder is not None:
            for param in self.decoder.parameters():
                param.requires_grad = False
    
    def unfreeze_decoder(self):
        if hasattr(self, 'decoder') and self.decoder is not None:
            for param in self.decoder.parameters():
                param.requires_grad = True
    
    def save_updated_codebooks(self, save_path: str):
        updated_codebooks = {}
        for i, (emb_layer, name) in enumerate(zip(self.codebook_embeddings, self.codebook_names)):
            updated_codebooks[name] = emb_layer.weight.data.cpu().numpy()
        
        with open(save_path, 'wb') as f:
            pickle.dump(updated_codebooks, f)


def create_rqvae_embedding(
    codebook_file: str,
    semantic_code_file: str,
    embedding_dim: int = 64,
    fusion_method: str = 'sum',
    learnable: bool = True
) -> RQVAECodebookEmbedding:
    return RQVAECodebookEmbedding(
        codebook_file=codebook_file,
        semantic_code_file=semantic_code_file,
        embedding_dim=embedding_dim,
        fusion_method=fusion_method,
        learnable=learnable
    )
