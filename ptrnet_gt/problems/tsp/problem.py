import os
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset


class TSP(object):

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)


class TSPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()
        self.filename = filename
        self.metadata = {
            "distribution": distribution,
            "size": size,
            "num_samples": num_samples,
            "offset": offset,
        }

        if filename is not None:
            suffix = Path(filename).suffix
            if suffix == '.pkl':
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                sliced = data[offset:offset + num_samples]
                self.data = [torch.as_tensor(row, dtype=torch.float32) for row in sliced]
                self.metadata.update({
                    "source_format": "pkl",
                    "dataset_path": str(filename),
                    "num_instances": len(self.data),
                })
            elif suffix == '.pt':
                payload = torch.load(filename, map_location='cpu')
                if isinstance(payload, dict):
                    coords = payload.get("coords")
                    self.metadata.update({k: v for k, v in payload.items() if k != "coords"})
                else:
                    coords = payload
                if coords is None:
                    raise ValueError(f"Dataset file {filename} does not contain 'coords'")
                coords = torch.as_tensor(coords, dtype=torch.float32)
                if coords.ndim != 3 or coords.size(-1) != 2:
                    raise ValueError(f"Expected coords with shape [N, n_nodes, 2], got {tuple(coords.shape)}")
                coords = coords[offset:offset + num_samples]
                self.data = [row.clone() for row in coords]
                self.metadata.update({
                    "source_format": "pt",
                    "dataset_path": str(filename),
                    "num_instances": len(self.data),
                    "size": int(coords.size(1)) if coords.numel() > 0 else self.metadata.get("size", size),
                })
            else:
                raise ValueError(f"Unsupported dataset format: {suffix}")
        else:
            self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for _ in range(num_samples)]
            self.metadata.update({
                "source_format": "generated",
                "num_instances": len(self.data),
            })

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


__all__ = ["TSP", "TSPDataset"]