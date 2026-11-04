import numpy as np
import torch

print('test')


# 1. Check the installed PyTorch version
print(f"PyTorch Version: {torch.__version__}")

# 2. Create and print a simple tensor (a basic operation)
x = torch.rand(5, 3)
print("\nTest Tensor (CPU):")
print(x)