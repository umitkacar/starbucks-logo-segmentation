from functools import cached_property, partial
from typing import Dict

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
from torch_optimizer import RAdam

from mobile_seg.loss import dice_loss, ce_loss
from mobile_seg.modules.net import MobileNetV2_unet
from mobile_seg.params import ModuleParams

from mylib.pytorch_lightning.base_module import PLBaseModule, StepResult
from mylib.torch.ensemble.ema import create_ema
from mylib.torch.optim.sched import flat_cos

# noinspection PyAbstractClass
class PLModule(PLBaseModule[MobileNetV2_unet]):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = MobileNetV2_unet(
            arch_name      = config["arch_name"],
            io_ratio       = config["io_ratio"],
            category       = config["category"],
            num_classes    = config["num_classes"],
            drop_rate      = config["drop_rate"],
            drop_path_rate = config["drop_path_rate"]
        )
        
        if config["category"] == "binary":
            self.criterion = dice_loss(scale=2) if config["io_ratio"] == "half" else dice_loss(scale=1)
        elif config["category"] == "multi":
            self.criterion = ce_loss(scale=2) if config["io_ratio"] == "half" else ce_loss(scale=1)

        if config["use_ema"]:
            self.ema_model = create_ema(self.model)

    def step(self, model: MobileNetV2_unet, batch) -> StepResult:
        X, y = batch
        y_hat = model.forward(X)
        # assert y.shape == y_hat.shape, f'{y.shape}, {y_hat.shape}'

        loss = self.criterion(y_hat, y)
        n_processed = len(y)

        return {
            'loss': loss,
            'n_processed': n_processed,
        }

    def configure_optimizers(self):
        params = self.model.parameters()

        if self.config["optim"] == 'adam':
            opt = Adam(
                params,
                lr           = self.config["lr"],
                weight_decay = self.config["weight_decay"],
            )
            sched = {
                'scheduler': OneCycleLR(
                    opt,
                    max_lr      = self.config["lr"],
                    total_steps = self.total_steps,
                ),
                'interval': 'step',
            }
        elif self.config["optim"] == 'radam':
            opt = RAdam(
                params,
                lr           = self.config["lr"],
                weight_decay = self.config["weight_decay"],
            )
            # noinspection PyTypeChecker
            sched = {
                'scheduler': LambdaLR(
                    opt,
                    lr_lambda = partial(
                        flat_cos,
                        total_steps = self.total_steps,
                    ),
                    verbose = True,
                ),
                'interval': 'step',
            }
        else:
            raise Exception

        return [opt], [sched]