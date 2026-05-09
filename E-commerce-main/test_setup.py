import torch
import importlib
import sys

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
print("MPS available:", torch.backends.mps.is_available())

missing = []
for package in ["transformers", "peft", "evaluate", "pandas", "PIL"]:
    try:
        module = importlib.import_module(package)
    except ImportError:
        missing.append(package)
        continue
    print(f"{package} version:", getattr(module, "__version__", "installed"))

if missing:
    print("Missing packages:", ", ".join(missing))
    sys.exit(1)
