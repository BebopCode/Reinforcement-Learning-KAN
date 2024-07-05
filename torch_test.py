import os
import torch

# Check if the environment variable is set
print("HSA_OVERRIDE_GFX_VERSION:", os.getenv("HSA_OVERRIDE_GFX_VERSION"))

# Check PyTorch version
print("PyTorch version:", torch.__version__)

# Check if ROCm is available (using the cuda module which is also used for ROCm)
if torch.cuda.is_available():
    print("ROCm is available. GPU is enabled.")
    print("Number of GPUs available:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} name:", torch.cuda.get_device_name(i))
else:
    print("ROCm is not available. Running on CPU.")