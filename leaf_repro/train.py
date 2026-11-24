"""Original LEAF optimizer/forward/loss plus resumable checkpoints and logging."""
import argparse
import json
import math
from pathlib import Path
import time

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .common import config, sha256, source_hashes, write_json
from .data import native, InstructionDataset, read_manifest, UPSTREAM
from load_config import load_yaml, build_model_config, build_train_config


class LoggedModule(native.Instruct_Tuning_Module):
    def __init__(self, *args, output, **kwargs):
        super().__init__(*args, **kwargs)
        self.output = Path(output)
        self.started = time.time()
        self.last_log_time = self.started

    def training_step(self, batch, batch_idx):
        # Identical objective and gradient to native.training_step.
        x, instruction, y, ds = batch
        emb, _ = self.model(x, instruction)
        logits = emb @ self.prototypes
        losses = F.cross_entropy(logits, y, reduction="none")
        loss = losses.mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=len(y))
        stats = torch.stack([torch.stack([(losses.detach() * (ds == i)).sum(), (ds == i).sum()]) for i in range(len(native.Targets))])
        stats = self.all_gather(stats).reshape(-1, len(native.Targets), 2).sum(0)
        for i, name in enumerate(native.Targets):
            if stats[i, 1] > 0:
                self.log("train_loss/" + name, stats[i, 0] / stats[i, 1], on_step=True, on_epoch=False, rank_zero_only=True)
        return loss

    def on_after_backward(self):
        if self.global_step == 1:
            groups = {"tower": self.model.tower, "instruction": self.model.qformer}
            audit = {}
            for name, mod in groups.items():
                grads = [p.grad for p in mod.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
                audit[name] = sum(float(g.detach().float().norm()) for g in grads)
                assert audit[name] > 0
            if self.global_rank == 0:
                write_json(self.output / "gradient_audit.json", audit)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_rank == 0 and (self.global_step <= 2 or self.global_step % 20 == 0):
            report = {"epoch": self.current_epoch + 1, "step": self.global_step, "loss": float(outputs["loss"]),
                      "elapsed_seconds": time.time() - self.started,
                      "learning_rates": [g["lr"] for g in self.trainer.optimizers[0].param_groups]}
            write_json(self.output / "progress.json", report)
            print(json.dumps(report), flush=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, instruction, y, ds = batch
        emb, _ = self.model(x, instruction)
        logits = emb @ self.prototypes
        losses = F.cross_entropy(logits, y, reduction="none")
        correct = (logits.argmax(-1) == y).float()
        self.log("loss", losses.mean(), on_epoch=True, sync_dist=True, batch_size=len(y))
        self.log("acc", correct.mean(), on_epoch=True, sync_dist=True, batch_size=len(y))
        for i in range(len(native.Targets)):
            mask = ds == i
            self.val_dataset_stats[i] += torch.stack([(losses * mask).sum(), (correct * mask).sum(), mask.sum()])

    def on_validation_epoch_start(self):
        self.val_dataset_stats = torch.zeros(len(native.Targets), 3, device=self.device)

    def on_validation_epoch_end(self):
        stats = self.all_gather(self.val_dataset_stats).reshape(-1, len(native.Targets), 3).sum(0)
        for i, name in enumerate(native.Targets):
            if stats[i, 2] > 0:
                self.log("valid_loss/" + name, stats[i, 0] / stats[i, 2], rank_zero_only=True)
                self.log("valid_global_acc/" + name, stats[i, 1] / stats[i, 2], rank_zero_only=True)
        super().on_validation_epoch_end()

    def on_train_epoch_end(self):
        # Full optimizer/scheduler state is saved by Lightning ModelCheckpoint.
        if self.global_rank == 0 and (self.current_epoch + 1) % 10 == 0:
            torch.save(self.model.state_dict(), self.output / f"leaf_epoch{self.current_epoch + 1:03d}.ckpt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    cfg = config()
    torch.set_num_threads(2)
    checkpoint = UPSTREAM / cfg["pretrain_checkpoint"]
    assert sha256(checkpoint) == cfg["pretrain_sha256"]
    manifest = read_manifest(smoke=args.smoke)
    out = Path(cfg["output_root"]) / ("smoke" if args.smoke else "train")
    out.mkdir(parents=True, exist_ok=True)
    upstream_cfg = load_yaml(UPSTREAM / "configs/LEAF_mpnet.yaml")
    model_cfg, train_cfg = build_model_config(upstream_cfg), build_train_config(upstream_cfg, "instruct")
    pl.seed_everything(cfg["seed"], workers=True)
    prototypes, key2idx, text2emb = native.build_prototype(native.Targets, model_cfg.text_emb_model_name)
    train = InstructionDataset(manifest, "train", key2idx, text2emb, 256 if args.smoke else None)
    valid = InstructionDataset(manifest, "valid", key2idx, text2emb, 32 if args.smoke else None)
    bs = cfg["batch_size_per_device"]
    steps = math.ceil(math.ceil(len(train) / len(cfg["devices"])) / bs)
    module = LoggedModule(model_cfg, train_cfg, prototypes, checkpoint, steps, seed=cfg["seed"], output=out)
    source = source_hashes()
    ckpt_callbacks = [ModelCheckpoint(dirpath=out / "checkpoints", filename="epoch{epoch:03d}", every_n_epochs=1,
                                     save_last=True, save_top_k=1),
                      ModelCheckpoint(dirpath=out / "selected", filename="best-{epoch:03d}", monitor="loss", mode="min", save_top_k=1)]
    trainer = pl.Trainer(accelerator="gpu", devices=cfg["devices"],
                         strategy=DDPStrategy(find_unused_parameters=True), precision=train_cfg.precision,
                         max_epochs=train_cfg.max_epochs, max_steps=3 if args.smoke else -1,
                         check_val_every_n_epoch=1 if args.smoke else train_cfg.check_val_every_n_epoch,
                         limit_val_batches=2 if args.smoke else 1.0,
                         num_sanity_val_steps=2, enable_progress_bar=False, enable_model_summary=False,
                         benchmark=True, deterministic=False, callbacks=ckpt_callbacks,
                         logger=CSVLogger(str(out), name="metrics", version=0), log_every_n_steps=1)
    if trainer.is_global_zero:
        write_json(out / "initial_state.json", {"config": cfg, "model": model_cfg.to_dict(), "optimizer": train_cfg.to_dict(),
                    "source_hashes": source, "train_windows": len(train), "valid_windows": len(valid),
                    "steps_per_epoch": steps, "global_batch": bs * len(cfg["devices"]), "prototype_labels": key2idx,
                    "parameters": sum(p.numel() for p in module.model.parameters()),
                    "trainable_parameters": sum(p.numel() for p in module.model.parameters() if p.requires_grad),
                    "tower_parameters": sum(p.numel() for p in module.model.tower.parameters()),
                    "manifest_sha256": sha256(Path(cfg["prepared_root"]) / ("smoke_manifest.json" if args.smoke else "manifest.json"))})
    loaders = [DataLoader(d, batch_size=bs, shuffle=i == 0, num_workers=cfg["workers_per_device"],
                          pin_memory=True, persistent_workers=True) for i, d in enumerate((train, valid))]
    last = out / "checkpoints/last.ckpt"
    if not args.resume and last.exists():
        raise FileExistsError(f"Use --resume for {last}")
    trainer.fit(module, train_dataloaders=loaders[0], val_dataloaders=loaders[1],
                ckpt_path=str(last) if args.resume and last.exists() else None)
    if trainer.is_global_zero:
        if source_hashes() != source:
            raise RuntimeError("Source changed during training")
        torch.save(module.model.state_dict(), out / "leaf_final.ckpt")
        write_json(out / "complete.json", {"step": trainer.global_step, "epochs": trainer.current_epoch,
                   "best_checkpoint": ckpt_callbacks[1].best_model_path, "seconds": time.time() - module.started,
                   "smoke": args.smoke, "source_hashes": source})


if __name__ == "__main__":
    main()
