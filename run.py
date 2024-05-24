
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
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = _datamodules["dataset_" + _config["dataset"]](_config)

    os.makedirs(_config["local_checkpoint_dir"], exist_ok=True)
    
    
    checkpoint_callback_epoch = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/total_loss",
        mode="min",
        save_last=True,
        auto_insert_metric_name=True,
        save_weights_only=False,#neu
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    model = FaceTTS(_config)

    model_summary_callback = pl.callbacks.ModelSummary(max_depth=2)

    callbacks = [checkpoint_callback_epoch, lr_callback, model_summary_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    
    trainer = pl.Trainer(
        accelerator="cpu", #new
        gpus=1, #new
        #resume_from_checkpoint=_config["resume_from"],
        #gpus=_config["num_gpus"], # this it was before
        num_nodes=_config["num_nodes"],
        #strategy=DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True), #strategy="ddp", #DAVOR DDPPLUGIN
        max_steps=max_steps,
        callbacks=callbacks,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=50,
        #flush_logs_every_n_steps=50,
        #weights_summary="top",
        #enable_model_summary=False, #neu
        val_check_interval=_config["val_check_interval"],

    )
    #NEU - VON MIR
    if _config["resume_from"]:
        checkpoint = torch.load(_config["resume_from"])
        
        print(checkpoint.keys())
        
        if 'global_step' not in checkpoint:
            checkpoint['global_step'] = 0  # Or set to an appropriate value based on your training
            torch.save(checkpoint, _config["resume_from"])
            print("Added 'global_step' to the checkpoint")
        if 'epoch' not in checkpoint:
            checkpoint['epoch'] = 0  # Or set to an appropriate value based on your training
            torch.save(checkpoint, _config["resume_from"])
            print("Added 'epoch' to the checkpoint")
        
        if 'optimizer_states' in checkpoint:
            print("Optimizer states found in the checkpoint.")
        else:
            print("Optimizer states not found in the checkpoint.")

    #################################################
    
    
    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm) #, ckpt_path=_config["resume_from"]

    else:
        trainer.test(model, datamodule=dm) #, ckpt_path=_config["resume_from"])