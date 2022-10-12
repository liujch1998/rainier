import os, sys
import argparse
import torch
import pytorch_lightning as pl
from imitation import Model

checkpoint_path = sys.argv[1]

pl_args = argparse.Namespace(model_type='t5-large')
pl_model = Model.load_from_checkpoint(checkpoint_path=checkpoint_path, args=pl_args)
t5_model = pl_model.model

torch.save(t5_model.state_dict(), f'../model/rainier-large_stageI.pth')

