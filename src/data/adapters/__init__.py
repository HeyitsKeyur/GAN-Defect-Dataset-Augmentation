"""Dataset adapters package for multi-dataset support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.adapters.base import BaseDatasetAdapter, DatasetImageRecord
from src.data.adapters.ceramic import CeramicAdapter
from src.data.adapters.soldef import SolDefAdapter

__all__ = [
    "BaseDatasetAdapter",
    "DatasetImageRecord",
    "CeramicAdapter",
    "SolDefAdapter",
    "get_adapter",
]


def get_adapter(dataset_name: str, **kwargs: Any) -> BaseDatasetAdapter:
    """Factory function to resolve and instantiate a dataset adapter.

    Args:
        dataset_name: "soldef" or "ceramic"
        **kwargs: Adapter-specific configuration arguments.
    """
    name = dataset_name.lower().strip()
    if name in ("soldef", "soldef_ai", "soldef-ai"):
        return SolDefAdapter(**kwargs)
    elif name in ("ceramic", "ceramic_defect", "ceramic-defect"):
        return CeramicAdapter(**kwargs)
    else:
        raise ValueError(
            f"Unsupported dataset name: '{dataset_name}'. Supported adapters: 'soldef', 'ceramic'"
        )
