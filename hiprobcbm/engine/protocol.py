"""Explicit optimization/selection policy for the controlled benchmark."""
import torch
from hiprobcbm.utils.checkpoint import RunCheckpoint


def optimizer_for(model, cfg, lr):
    """Optimizer per protokol. ``lr_ratio`` memberi parameter selain backbone
    lr * lr_ratio, seperti grup ``params_new`` ProbCBM (main.py; config_exp.yaml
    lr 1e-3, lr_ratio 10). Parameter beku tidak dimasukkan, seperti ProbCBM
    yang hanya mengoptimasi ``params_to_train``."""
    name = cfg.get("optimizer", "adam").lower()
    decay = cfg.get("weight_decay", 0.0) or 0.0
    ratio = cfg.get("lr_ratio", 1.0)
    backbone = getattr(model, "backbone", None)
    backbone_ids = {id(p) for p in backbone.parameters()} if backbone is not None else set()
    trainable = [p for p in model.parameters() if p.requires_grad]
    old = [p for p in trainable if id(p) in backbone_ids]
    new = [p for p in trainable if id(p) not in backbone_ids]
    params = [group for group in ({"params": old, "lr": lr}, {"params": new, "lr": lr * ratio}) if group["params"]]
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=cfg.get("momentum", .9), weight_decay=decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=decay)
    if name == "adamp":
        from adamp import AdamP
        return AdamP(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=decay)
    raise ValueError(f"Unknown optimizer: {name}")


def train_concept_labels(loader) -> torch.Tensor:
    """Label konsep induk split train tanpa mengonsumsi RNG DataLoader."""
    dataset = loader.dataset
    if hasattr(dataset, "concepts"):
        return torch.as_tensor(dataset.concepts).float()
    if hasattr(dataset, "records"):  # CUB: label ada di pkl, tanpa membuka gambar
        return torch.tensor([r["attribute_label"] for r in dataset.records], dtype=torch.float32)
    return torch.stack([torch.as_tensor(dataset[i]["concepts"]).float() for i in range(len(dataset))])


def clip_gradients(model, cfg) -> None:
    """``clip_grad_norm_`` ProbCBM (run_epoch.py; config_exp.yaml clip_grad_max_norm 2.0)."""
    max_norm = cfg.get("clip_grad_norm")
    if max_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def set_backbone_trainable(model, trainable: bool) -> None:
    """Warm epoch ProbCBM (run_epoch.py): ``cnn_module`` requires_grad=False
    selama ``warm_epochs`` pertama; BatchNorm tetap mengikuti model.train()."""
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(trainable)


def checkpoint_for(directory, name, model, optimizer, identity, epochs, cfg, resume):
    scheduler = None
    policy = cfg.get("scheduler", "none")
    if policy == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=.1, patience=cfg.get("lr_patience", 10))
    elif policy == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif policy != "none":
        raise ValueError(f"Unknown scheduler: {policy}")
    return RunCheckpoint(directory, name, model, optimizer, identity, epochs, resume,
                         scheduler=scheduler, patience=cfg.get("patience"))
