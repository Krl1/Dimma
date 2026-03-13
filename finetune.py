import argparse
import json
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from src.wandb_logger import WandbLoggerWithCache

from src.datasets import LOLDataModule, FSDDataModule, CECDataModule  # noqa: I900
from src.models import LitDimma  # noqa: I900

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/FS-Dark/stage2/6shot-fsd-ft.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)

    if cfg.dataset.name == "LOL":
        dm = LOLDataModule(config=cfg.dataset)
    elif cfg.dataset.name == "FSD":
        dm = FSDDataModule(config=cfg.dataset)
    elif cfg.dataset.name == "CEC":
        dm = CECDataModule(config=cfg.dataset)

    # model = LitDimma(config=cfg)
    model = LitDimma.load_from_checkpoint(cfg.model.checkpoint, config=cfg, weights_only=False)

    callbacks = [
        pl.callbacks.progress.TQDMProgressBar(),
        ModelCheckpoint(
            monitor="val/psnr",
            mode="max",
            save_top_k=cfg.logger.save_top_k,
            save_last=False,
            auto_insert_metric_name=False,
            filename=cfg.name,
            dirpath=f'{cfg.logger.checkpoint_dir}/{cfg.name}',
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # uncomment to use wandb
    logger = WandbLoggerWithCache(
        entity="biocam", 
        project="Dimma-finetune", 
        name=cfg.name,
        save_dir="logs",
        tags=[cfg.dataset.name, cfg.model.head, "finetune"],
        group=f"stage2-{cfg.dataset.name}"
    )

    trainer = pl.Trainer(
        accelerator=cfg.device,
        devices=1,
        callbacks=callbacks,
        logger=logger, # use wandb
        max_steps=cfg.iter,
        check_val_every_n_epoch=None,
        val_check_interval=cfg.eval_freq,
        log_every_n_steps=10,
    )

    print(f"🚀 Starting finetuning: {cfg.name}")
    trainer.validate(model, datamodule=dm)

    trainer.fit(model, dm)
    print(f"✅ Finetuning finished. Best model: {trainer.checkpoint_callback.best_model_path}")

    # load best model
    model = LitDimma.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, config=cfg, weights_only=False
    )

    # test
    output = trainer.test(model, datamodule=dm)

    # save results as json
    with open(f'{cfg.logger.checkpoint_dir}/{cfg.name}/results.json', 'w') as f:
        json.dump(output[0], f, indent=4)

