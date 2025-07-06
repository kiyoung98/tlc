import torch

from torch.optim import Adam
from mlcolvar.cvs import AutoEncoderCV

from ..util.constant import *


class TAE(AutoEncoderCV):
    def __init__(self, *args, **kwargs):
        reference_frame = torch.load(kwargs.pop('reference_frame'))['xyz'].reshape(-1, 3)
        super().__init__(*args, **kwargs)
        self.cv_normalize = False
        self.register_buffer("reference_frame", reference_frame)
        self.optimizer = Adam(self.parameters(), lr=1e-3)

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ):
        optimizer = self.optimizer
        optimizer.step(closure=optimizer_closure)

    def backward(self, loss):
        loss.backward(retain_graph=True)
    