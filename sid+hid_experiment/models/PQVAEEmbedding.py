
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from models.quantized_embedding_base import QuantizedCodebookEmbeddingBase


class PQVAEEmbedding(QuantizedCodebookEmbeddingBase):

    def __init__(
        self,
        *,
        codebook_file: str,
        semantic_code_file: str,
        embedding_dim: int,
        padding_idx: int = 0,
        fusion_method: str = "mean",
        learnable: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        codebook_path = Path(codebook_file)
        codes_path = Path(semantic_code_file)

        payload = self.load_codebook_payload(str(codebook_path))
        codebook_names, codebook_tensors = self._extract_codebooks(payload)
        semantic_codes = self.load_semantic_codes(str(codes_path))

        super().__init__(
            codebook_tensors=codebook_tensors,
            codebook_names=codebook_names,
            semantic_codes=semantic_codes,
            embedding_dim=embedding_dim,
            fusion_method=fusion_method,
            learnable=learnable,
            padding_idx=padding_idx,
            device=device,
            metadata=payload.get("metadata", {}),
        )


    @staticmethod
    def _extract_codebooks(payload: Dict) -> (List[str], List[torch.Tensor]):
        codebooks = payload.get("codebooks")
        if not codebooks:
            raise ValueError("PQVAE codebook payload missing 'codebooks' key.")

        names = sorted(codebooks.keys(), key=_extract_index)
        tensors = [torch.from_numpy(np.asarray(codebooks[name])) for name in names]
        return names, tensors


def _extract_index(name: str) -> int:
    for token in name.split("_"):
        if token.isdigit():
            return int(token)
    return 0


