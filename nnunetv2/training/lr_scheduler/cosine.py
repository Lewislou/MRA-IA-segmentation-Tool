from math import cos, pi

from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, eta_min: float = 0.0,
                 current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max(1, int(max_steps))
        self.eta_min = eta_min
        self.ctr = 0
        super().__init__(optimizer, last_epoch=current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        t = min(max(int(current_step), 0), self.max_steps)
        new_lr = self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (
            1.0 + cos(pi * t / self.max_steps)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
