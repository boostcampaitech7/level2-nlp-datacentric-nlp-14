import os

import torch

SEED = 456
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
