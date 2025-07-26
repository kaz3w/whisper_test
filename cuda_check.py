#!/usr/bin/env python3
import torch
print(torch.cuda.is_available())  # TrueならGPU使用可能
print(torch.cuda.get_device_name(0))  # 使用中のGPU名

