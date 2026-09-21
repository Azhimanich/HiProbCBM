"""Explicit optimization/selection policy for the controlled benchmark."""
import torch
from hiprobcbm.utils.checkpoint import RunCheckpoint


def optimizer_for(model, cfg, lr):
    name = cfg.get("optimizer", "adam").lower()
    params = model.parameters()
    decay = cfg.get("weight_decay", 0.0)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=cfg.get("momentum", .9), weight_decay=decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=decay)
    if name == "adamp":
        from adamp import AdamP
        return AdamP(params, lr=lr, weight_decay=decay)
    raise ValueError(f"Unknown optimizer: {name}")


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
