
from logging import getLogger, FileHandler
from time import time
from pathlib import Path
import json

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from mylib.pytorch_lightning.logging import configure_logging

from data import DataModule
from model import PLModule

if __name__ == '__main__':

    with open(f'params/config.json') as f:
        config = json.load(f)
    config["save_dir"] = Path('../experiments')

    configure_logging()
    pl.seed_everything(config["seed"])

    tb_logger = pl.loggers.TensorBoardLogger(
        config["save_dir"],
        name=f'mobile_seg',
        version=str(int(time())),
    )

    log_dir = Path(tb_logger.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = getLogger('lightning')
    logger.addHandler(FileHandler(log_dir / 'train.log'))
    logger.info(config)

    trainer = pl.Trainer(
        max_epochs             = config["epoch"],
        gpus                   = config["gpus"],
        tpu_cores              = config["num_tpu_cores"],
        precision              = config["precision"],
        weights_save_path      = config["save_dir"],
        resume_from_checkpoint = config["resume_from_checkpoint"],
        logger                 = tb_logger,
        checkpoint_callback    = True,
        deterministic          = True,
        benchmark              = True,
        callbacks=[
            ModelCheckpoint(
                monitor='ema_0_loss' if config["use_ema"] else 'val_0_loss',
                save_last=True,
                verbose=True,
            )],   
    )
    
    net = PLModule(config)
    dm = DataModule(config)
    trainer.fit(net, datamodule=dm)