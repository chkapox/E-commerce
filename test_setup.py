import torch
import transformers

print("Torch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("Transformers version:", transformers.__version__)
