import gc
import math

from pytorch_lightning.callbacks import Callback
from torch.utils.data import ConcatDataset, DataLoader


class _BestValidationEpoch(Callback):
    def __init__(self):
        self.best_epoch = None
        self.best_metric = None

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        metric = trainer.callback_metrics.get("val_metric_mean")
        if hasattr(metric, "item"):
            metric = metric.item()
        if metric is None or not math.isfinite(float(metric)):
            return
        if self.best_metric is None or metric > self.best_metric:
            self.best_epoch = epoch
            self.best_metric = metric


def _refit_loader(train_loader, val_loader):
    kwargs = {
        "batch_size": train_loader.batch_size,
        "shuffle": True,
        "num_workers": train_loader.num_workers,
        "pin_memory": train_loader.pin_memory,
        "drop_last": train_loader.drop_last,
        "collate_fn": train_loader.collate_fn,
    }
    if train_loader.num_workers:
        kwargs["persistent_workers"] = train_loader.persistent_workers
        kwargs["worker_init_fn"] = train_loader.worker_init_fn
    return DataLoader(ConcatDataset([train_loader.dataset, val_loader.dataset]), **kwargs)


def run_mtt_selection_and_refit(
    config,
    train_loader,
    val_loader,
    test_loader,
    logger,
    model_factory,
    trainer_factory,
    early_stopping_factory,
    seed_fn,
):
    best_epoch = _BestValidationEpoch()
    selection_trainer = trainer_factory(
        max_epochs=config.training.epochs,
        logger=logger,
        accelerator="auto",
        devices="auto",
        callbacks=[
            early_stopping_factory(
                monitor="val_metric_mean", patience=config.patience, mode="max", verbose=True
            ),
            best_epoch,
        ] if config.patience > 0 else [best_epoch],
        deterministic=True,
        enable_checkpointing=False,
    )
    selection_model = model_factory(config)
    selection_trainer.fit(selection_model, train_loader, val_loader)
    if best_epoch.best_epoch is None:
        raise RuntimeError("MTT selection observed no finite val_metric_mean")

    refit_epochs = best_epoch.best_epoch + 1
    refit_loader = _refit_loader(train_loader, val_loader)
    audit = {
        "mtt_selection_best_epoch": best_epoch.best_epoch,
        "mtt_refit_epochs": refit_epochs,
        "mtt_n_train": len(train_loader.dataset),
        "mtt_n_val": len(val_loader.dataset),
        "mtt_n_refit": len(refit_loader.dataset),
        "mtt_n_test": len(test_loader.dataset),
    }
    logger.log_hyperparams(audit)
    print("MTT selection/refit provenance: " + ", ".join(f"{key}={value}" for key, value in audit.items()))

    del selection_model
    del selection_trainer
    gc.collect()
    seed_fn(config.seed, workers=True)
    refit_model = model_factory(config)
    refit_trainer = trainer_factory(
        max_epochs=refit_epochs,
        logger=logger,
        accelerator="auto",
        devices="auto",
        deterministic=True,
        enable_checkpointing=False,
    )
    refit_trainer.fit(refit_model, refit_loader)
    refit_trainer.test(refit_model, test_loader)
    return {"best_epoch": best_epoch.best_epoch, "refit_epochs": refit_epochs, **audit}


def run_standard_training(config, model, train_loader, val_loader, test_loader, logger, trainer_factory, early_stopping_factory):
    early_stop = early_stopping_factory(
        monitor="val_metric_mean", patience=config.patience, mode="max", verbose=True
    )
    trainer = trainer_factory(
        max_epochs=config.training.epochs,
        logger=logger,
        accelerator="auto",
        devices="auto",
        callbacks=[early_stop] if config.patience > 0 else None,
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
