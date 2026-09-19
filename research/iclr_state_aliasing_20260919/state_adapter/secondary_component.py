"""Decoder-adaptation extension; leaves the frozen-host implementation intact."""
import torch
from component_adapter import Model as FrozenModel


class Model(FrozenModel):
    def __init__(self,aware=True,kind='icam'):
        super().__init__(aware,kind)
        if kind=='icam':
            self.host.net.decoder.requires_grad_(True)
        else:
            for name,p in self.host.net.named_parameters():
                if not name.startswith(('init_embed.','embedder.')): p.requires_grad_(True)
        self.decoder_names=[name for name,p in self.host.named_parameters() if p.requires_grad]

    def trainable_parameters(self): return [p for p in self.parameters() if p.requires_grad]

    def capture(self):
        return dict(residual={k:v.detach().cpu().clone() for k,v in self.residual.state_dict().items()},
                    decoder={name:p.detach().cpu().clone() for name,p in self.host.named_parameters() if p.requires_grad})

    def restore(self,ck):
        self.residual.load_state_dict(ck['residual'])
        current=dict(self.host.named_parameters())
        with torch.no_grad():
            for name,value in ck['decoder'].items(): current[name].copy_(value)
