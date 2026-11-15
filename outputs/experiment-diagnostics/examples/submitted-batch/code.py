import torch
samples = [[1, 2, 3, 4], [5, 6]]
batch = torch.stack([torch.tensor(s) for s in samples])
