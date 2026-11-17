#!/usr/bin/env -S colab run --gpu T4 --timeout 0
"""Confirm the Colab runtime rented for a CASMI experiment has a CUDA GPU."""

import torch

available = torch.cuda.is_available()
print("cuda", available)
if available:
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
