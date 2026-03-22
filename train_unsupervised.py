import argparse
import json
import pytorch_lightning as pl
from omegaconf import OmegaConf

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from src.wandb_logger import WandbLoggerWithCache

from src.datasets import MixHQDataModule, CECDataModule  # noqa: I900
from src.models import LitDimma # noqa: I900


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/FS-Dark/stage1/6shot-fsd.yaml")
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(conf))

    pl.seed_everything(conf.seed)

    if conf.dataset.name == "MixHQ":
        dm = MixHQDataModule(config=conf.dataset)
    elif conf.dataset.name == "CEC":
        dm = CECDataModule(config=conf.dataset)

    model = LitDimma(config=conf)

    callbacks = [
        pl.callbacks.progress.TQDMProgressBar(),
        ModelCheckpoint(
            monitor="val/psnr",
            mode="max",
            save_top_k=conf.logger.save_top_k,
            save_last=False,
            auto_insert_metric_name=False,
            filename=conf.name,
            dirpath=f'{conf.logger.checkpoint_dir}/{conf.name}',
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # uncomment to use wandb
    logger = WandbLoggerWithCache(
        entity="biocam", 
        project="Dimma-unsupervised",
        name=conf.name,
        save_dir="logs",
        tags=[conf.dataset.name, conf.model.head, "unsupervised"],
        group=f"stage1-{conf.dataset.name}"
    )

    trainer = pl.Trainer(
        accelerator=conf.device,
        devices=1,
        callbacks=callbacks,
        logger=logger, # use wandb
        max_steps=conf.iter,
        val_check_interval=conf.eval_freq,
        log_every_n_steps=10, 
    )
    print(f"🚀 Starting training: {conf.name}")
    trainer.fit(model, dm)
    print(f"✅ Training finished. Best model: {trainer.checkpoint_callback.best_model_path}")

    # load best model
    model = LitDimma.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, config=conf, weights_only=False
    )

    # test
    output = trainer.test(model, datamodule=dm)

    # save results as json
    with open(f'{conf.logger.checkpoint_dir}/{conf.name}/results.json', 'w') as f:
        json.dump(output[0], f, indent=4)