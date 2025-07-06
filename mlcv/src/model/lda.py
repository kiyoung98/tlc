import torch
from mlcolvar.core import LDA

class LDA(LDA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.postprocessing = None
        
    def forward(self, x):
        x = torch.matmul(x, self.evecs)
        if self.postprocessing:
            x = self.postprocessing(x)
        
        return x
    
    def to(self, device):
        self.device = device
        if self.evals is not None:
            self.evals = self.evals.to(device)
        if self.evecs is not None:
            self.evecs = self.evecs.to(device)
        if self.S_b is not None:
            self.S_b = self.S_b.to(device)
        if self.S_w is not None:
            self.S_w = self.S_w.to(device)
        
        if self.postprocessing:
            self.postprocessing.to(device)
        
        return self