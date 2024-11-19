import torch
import torchvision

torch.set_default_tensor_type('torch.FloatTensor')  # Force CPU tensors

print(torch.__version__)
print(torchvision.__version__)
