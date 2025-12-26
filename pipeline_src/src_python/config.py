#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 07:01:52 2025

@author: jctourtellotte
"""

import torch
import numpy as np

# Determine the device to use
# "one-liner": torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# using multiple lines in the event additional settings are device dependent

# dfloat_torch = torch.float
# dfloat_np = float

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("--- Config: CUDA is available. Using GPU. ---")
#    print(f"Default torch dtype set to: {dfloat_torch}")

# elif torch.backends.mps.is_available():

#     DEVICE = torch.device("mps")
#     print("--- Config: CUDA not available. Using MPS. ---")
#     dfloat_torch = torch.float32
#     dfloat_np = np.float32
#     print(f"Default torch dtype set to: {dfloat_torch}")
    

else:

    DEVICE = torch.device("cpu")
    print("--- Config: CUDA and MPS not available. Using CPU. ---")
#    print(f"Default torch dtype set to: {dfloat_torch}")
