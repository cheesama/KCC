from KCC import trainer

import os, sys
import torch

torch.multiprocessing.set_sharing_strategy('file_system')

trainer.train(
    file_path="nlu.md",
    batch_size=128,
    intent_optimizer_lr=5e-5,
    entity_optimizer_lr=5e-5,
    gpu_num=0,
    epochs=3
)
