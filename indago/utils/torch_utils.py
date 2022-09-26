import torch

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).to(DEVICE)


def from_numpy_no_device(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs)


def to_numpy(tensor):
    return tensor.to("cpu").detach().numpy()
