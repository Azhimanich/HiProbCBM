"""HiProbCBM - Hierarchical Probabilistic Concept Bottleneck Model.

Implementasi mengacu pada usulan penelitian S2 "Model Concept Bottleneck
Hierarkis Probabilistik melalui Agregasi Subkonsep Berbasis Learned
Attention" (Azhima, 2026), yang mengintegrasikan:

- Representasi konsep probabilistik dari Probabilistic Concept Bottleneck
  Model / ProbCBM (Kim dkk., ICML 2023).
- Automatic hierarchical subconcept discovery dari Hierarchical Concept
  Embedding Model / HiCEM (Hill dkk., ICLR 2026).

Setiap modul inti menyertakan referensi ke subbab proposal (Bab III/IV)
tempat formulasinya didefinisikan, agar kode dan naskah tesis tetap
tertelusuri satu sama lain.
"""

__version__ = "0.1.0"
