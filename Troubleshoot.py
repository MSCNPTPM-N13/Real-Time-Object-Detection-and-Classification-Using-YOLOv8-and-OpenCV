import torch
import torchvision

torch.set_default_device("cpu")  # Force CPU tensors

print(torch.__version__)
print(torchvision.__version__)
