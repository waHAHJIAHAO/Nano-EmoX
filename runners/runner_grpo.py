import logging

from nano_emox.common.registry import registry
from nano_emox.runners.runner_base import RunnerBase


@registry.register_runner("runner_grpo")
class RunnerGRPO(RunnerBase):

    def train_epoch(self, epoch):
        self.model.train()

        return self.task.train_epoch_grpo(
            epoch=epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
            run_cfg=self.config.run_cfg,
        )


