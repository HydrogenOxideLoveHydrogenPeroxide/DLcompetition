#文件集成部分
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"