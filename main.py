# main.py
from lightning.pytorch.cli import LightningCLI
import torch
import model.lit_asc
import data.data_module
import util
import model.backbones

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

cli = LightningCLI()
