"""Integrasi penuh Bab IV.5: Tahap 1 -> automatic subconcept discovery ->
Tahap 2 -> concept intervention, memakai backbone kecil & data dummy supaya
cepat dijalankan di CPU tanpa dataset asli.
"""

import torch

from hiprobcbm.intervention import build_subconcept_intervention_tensors
from hiprobcbm.losses import hiprobcbm_stage1_loss, hiprobcbm_stage2_loss
from hiprobcbm.models.hiprobcbm import HiProbCBMStage1, HiProbCBMStage2
from hiprobcbm.models.subconcept_discovery import build_pseudo_hierarchy

BATCH, IMG, NUM_CONCEPTS, NUM_CLASSES, D_C = 4, 224, 4, 3, 8


def test_stage1_forward_and_loss():
    model = HiProbCBMStage1("resnet18", num_concepts=NUM_CONCEPTS, concept_dim=D_C, pretrained=False)
    x = torch.randn(BATCH, 3, IMG, IMG)
    c = torch.randint(0, 2, (BATCH, NUM_CONCEPTS)).float()

    out = model(x, n_samples=3)
    assert out.mu.shape == (BATCH, NUM_CONCEPTS, D_C)
    assert out.sigma2.shape == (BATCH, NUM_CONCEPTS, D_C)
    assert out.concept_probs.shape == (BATCH, NUM_CONCEPTS)

    loss, components = hiprobcbm_stage1_loss(out.concept_probs, c, out.kl_loss)
    assert loss.item() >= 0
    loss.backward()  # pastikan graph valid end-to-end


def test_full_pipeline_stage1_to_stage2():
    torch.manual_seed(0)

    # --- Tahap 1 (dummy, tanpa training penuh - cukup forward untuk mu) ---
    stage1 = HiProbCBMStage1("resnet18", num_concepts=NUM_CONCEPTS, concept_dim=D_C, pretrained=False)
    n_train = 24
    x_train = torch.randn(n_train, 3, IMG, IMG)

    with torch.no_grad():
        mu_all, probs_all = [], []
        for i in range(0, n_train, BATCH):
            out = stage1(x_train[i : i + BATCH], n_samples=3)
            mu_all.append(out.mu)
            probs_all.append(out.concept_probs)
        mu_all = torch.cat(mu_all, dim=0)
        probs_all = torch.cat(probs_all, dim=0)

    # --- Automatic subconcept discovery (Bab IV.5.1.3-4.5.1.4) ---
    hierarchy = build_pseudo_hierarchy(
        mu_all=mu_all,
        concept_probs=probs_all,
        concept_names=[f"concept_{i}" for i in range(NUM_CONCEPTS)],
        sae_variant="kl",
        sae_kwargs={"latent_dim": 16},
        train_kwargs={"n_epochs": 2, "batch_size": 8},
    )
    subconcepts_per_concept = hierarchy.subconcepts_per_concept
    assert len(subconcepts_per_concept) == NUM_CONCEPTS

    # --- Tahap 2 (Bab IV.5.2) ---
    stage2 = HiProbCBMStage2(
        "resnet18",
        num_classes=NUM_CLASSES,
        subconcepts_per_concept=subconcepts_per_concept,
        concept_dim=D_C,
        pretrained=False,
        n_mc_samples_train=3,
        n_mc_samples_eval=5,
    )

    x = torch.randn(BATCH, 3, IMG, IMG)
    y = torch.randint(0, NUM_CLASSES, (BATCH,))
    c_parent = torch.randint(0, 2, (BATCH, NUM_CONCEPTS)).float()

    stage2.train()
    out = stage2(x)

    assert out.mu_parent.shape == (BATCH, NUM_CONCEPTS, D_C)
    assert out.attention.shape[0] == BATCH
    assert torch.allclose(out.attention.sum(dim=-1), torch.ones(BATCH, NUM_CONCEPTS), atol=1e-4)
    assert out.task_probs.shape == (BATCH, NUM_CLASSES)
    assert torch.allclose(out.task_probs.sum(dim=-1), torch.ones(BATCH), atol=1e-4)

    k_max = out.sub_concept_probs.shape[-1]
    sub_mask = stage2.subconcept_predictor.mask.unsqueeze(0).expand(BATCH, -1, -1)
    sub_labels = torch.zeros(BATCH, NUM_CONCEPTS, k_max)

    loss, components = hiprobcbm_stage2_loss(
        task_logits_per_sample=out.task_logits_per_sample,
        task_labels=y,
        parent_concept_probs=out.parent_concept_probs,
        parent_concept_labels=c_parent,
        sub_concept_probs=out.sub_concept_probs,
        sub_concept_mask=sub_mask,
        sub_concept_labels=sub_labels,
        kl_loss=out.kl_loss,
    )
    assert loss.item() >= 0
    loss.backward()


def test_stage2_uniform_aggregator_ablation_a1():
    """HiProbCBM-A1 (Tabel 4.8): use_attention=False harus tetap forward
    normal, hanya dengan bobot seragam."""
    subconcepts_per_concept = [2, 2, 0, 3]
    stage2 = HiProbCBMStage2(
        "resnet18", num_classes=NUM_CLASSES, subconcepts_per_concept=subconcepts_per_concept,
        concept_dim=D_C, pretrained=False, use_attention=False,
    )
    x = torch.randn(BATCH, 3, IMG, IMG)
    out = stage2(x)
    assert out.task_probs.shape == (BATCH, NUM_CLASSES)


def test_stage2_concept_intervention_changes_output():
    """Bab IV.5.4: intervensi subkonsep harus mengubah mu_parent (dan pada
    umumnya task_probs) dibanding forward tanpa intervensi."""
    torch.manual_seed(1)
    subconcepts_per_concept = [3, 2, 2, 1]
    stage2 = HiProbCBMStage2(
        "resnet18", num_classes=NUM_CLASSES, subconcepts_per_concept=subconcepts_per_concept,
        concept_dim=D_C, pretrained=False,
    )
    stage2.eval()
    x = torch.randn(BATCH, 3, IMG, IMG)

    with torch.no_grad():
        out_before = stage2(x)

        k_max = stage2.subconcept_predictor.k_max
        gt_sub_labels = torch.randint(0, 2, (BATCH, len(subconcepts_per_concept), k_max)).float()
        anchor_pos = torch.ones(D_C) * 5.0   # anchor jauh dari distribusi awal supaya efeknya kelihatan
        anchor_neg = torch.ones(D_C) * -5.0
        intervene_mask = stage2.subconcept_predictor.mask.clone()  # intervensi semua subkonsep valid

        target, mask = build_subconcept_intervention_tensors(
            stage2, gt_sub_labels, anchor_pos, anchor_neg, intervene_mask
        )
        out_after = stage2(x, subconcept_intervention=target, subconcept_intervention_mask=mask)

    assert not torch.allclose(out_before.mu_parent, out_after.mu_parent), (
        "mu_parent seharusnya berubah setelah concept intervention"
    )


if __name__ == "__main__":
    test_stage1_forward_and_loss()
    test_full_pipeline_stage1_to_stage2()
    test_stage2_uniform_aggregator_ablation_a1()
    test_stage2_concept_intervention_changes_output()
    print("OK: test_hiprobcbm_end_to_end")
