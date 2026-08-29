from hiprobcbm.data.base import ConceptDataset, DatasetSplits
from hiprobcbm.data.cub import CUBConceptDataset
from hiprobcbm.data.kitchens import PseudoKitchensDataset

DATASET_REGISTRY = {
    "cub": CUBConceptDataset,
    "kitchens": PseudoKitchensDataset,
}


def build_dataset(name: str, **kwargs) -> ConceptDataset:
    try:
        cls = DATASET_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Dataset '{name}' tidak dikenal. Pilihan: {list(DATASET_REGISTRY)}"
        ) from exc
    return cls(**kwargs)


__all__ = [
    "ConceptDataset",
    "DatasetSplits",
    "CUBConceptDataset",
    "PseudoKitchensDataset",
    "build_dataset",
    "DATASET_REGISTRY",
]
