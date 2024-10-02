
import pytorch_lightning as pl

#from pytorch_lightning.plugins import DDPPlugin #EIGENTLICH DRIN

from pytorch_lightning.strategies import DDPStrategy#new
#from pytorch_lightning.callbacks import ModelSummary #new
import os
import copy

from config import ex
from model.face_tts import FaceTTS
from data import _datamodules
import torch

#von mir
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
import gc
gc.collect()
torch.cuda.empty_cache()

#from pytorch_lightning.callbacks import ModelCheckpoint
@ex.automain
def main(_config):
    pl.seed_everything(_config["seed"])

    dm = _datamodules["dataset_" + _config["dataset"]](_config)

    # Load the model architecture
    model = FaceTTS(_config)

    # Load the model's state dictionary
    checkpoint = torch.load(_config["resume_from"])
    model.load_state_dict(checkpoint['state_dict'])

    # Initialize the trainer
    trainer = pl.Trainer(
        accelerator="cpu", 
        num_nodes=_config["num_nodes"],
        accumulate_grad_batches=_config["batch_size"] // (_config["per_gpu_batchsize"] * _config["num_gpus"] * _config["num_nodes"]),
        log_every_n_steps=50,
        val_check_interval=_config["val_check_interval"],
        resume_from_checkpoint=None,  # Skip restoring optimizer states
        max_steps=_config["max_steps"],
        callbacks=[pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            monitor="val/total_loss",
            mode="min",
            save_last=True,
            auto_insert_metric_name=True,
        )]
    )

    # Train or test the model
    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)