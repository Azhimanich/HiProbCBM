"""Reimplementasi baseline (CBM, CEM, ProbCBM, HiCEM) untuk Tabel 4.7 -
seluruhnya memakai `backbones.build_backbone` dan
`classifier.LinearSoftmaxClassifier` YANG SAMA dengan HiProbCBM
(Keputusan Desain Eksperimen #1), supaya perbedaan hasil dapat
diatribusikan pada representasi konsep, bukan komponen di luar itu.

Ini adalah tulisan ulang mandiri berdasarkan formulasi pada paper asli
masing-masing (Koh dkk. 2020; Zarlenga dkk. 2022; Kim dkk. 2023; Hill dkk.
2026), BUKAN salinan kode dari repo resmi `ejkim47/prob-cbm` atau
`OscarPi/cem-concept-discovery` - keduanya hanya dipelajari sebagai
rujukan arsitektur (lihat README, bagian "Kredit & Referensi").
"""

from hiprobcbm.models.baselines.cbm import ConceptBottleneckModel
from hiprobcbm.models.baselines.cem import ConceptEmbeddingModel
from hiprobcbm.models.baselines.hicem import HierarchicalConceptEmbeddingModel
from hiprobcbm.models.baselines.probcbm import ProbabilisticConceptBottleneckModel

BASELINE_REGISTRY = {
    "cbm": ConceptBottleneckModel,
    "cem": ConceptEmbeddingModel,
    "probcbm": ProbabilisticConceptBottleneckModel,
    "hicem": HierarchicalConceptEmbeddingModel,
}

__all__ = [
    "ConceptBottleneckModel",
    "ConceptEmbeddingModel",
    "ProbabilisticConceptBottleneckModel",
    "HierarchicalConceptEmbeddingModel",
    "BASELINE_REGISTRY",
]
