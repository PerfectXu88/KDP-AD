if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import threading
import shutil
from diffusion.workspace.base_workspace import BaseWorkspace
from diffusion.policy.metadrive_diffusion_transformer_policy import MetadriveDiffusionTransformerPolicy
from diffusion.dataset.base_dataset import BaseImageDataset
from diffusion.env_runner.base_image_runner import BaseImageRunner
from diffusion.common.checkpoint_util import TopKCheckpointManager
from diffusion.common.json_logger import JsonLogger
from diffusion.common.pytorch_util import dict_apply, optimizer_to
from diffusion.model.diffusion.ema_model import EMAModel
from diffusion.model.common.lr_scheduler import get_scheduler
from typing import List
from diffusion.dataset.multitask_dataset import MultiDataLoader
from itertools import zip_longest
OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainMetadriveDiffusionTransformerWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: MetadriveDiffusionTransformerPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: MetadriveDiffusionTransformerPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:   
            lastest_ckpt_path = pathlib.Path("")
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        datasets: List[BaseImageDataset] = []
        for i in range(cfg.task_num):
            datasets.append(hydra.utils.instantiate(cfg[f'task{i}'].dataset))
        
        assert isinstance(datasets[0], BaseImageDataset)
        train_dataloaders = []
        normalizers=[]
        for dataset in datasets:
            train_dataloaders.append(DataLoader(dataset, **cfg.dataloader))
            normalizers.append(dataset.get_normalizer())
        max_train_dataloader_len = max([len(train_dataloader) for train_dataloader in train_dataloaders])
        for train_dataloader in train_dataloaders:
            print("Length of train_dataloader: ", len(train_dataloader))
        multi_traindataloader=MultiDataLoader(train_dataloaders)
        multi_traindataloader.get_memory_usage()
        # configure validation dataset
        val_datasets=[]
        for dataset in datasets:
            val_datasets.append(dataset.get_validation_dataset())
       
        val_dataloaders = []
        for val_dataset in val_datasets:
            val_dataloaders.append(DataLoader(val_dataset, **cfg.val_dataloader))


        self.model.set_normalizer(normalizers)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizers)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                max_train_dataloader_len * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        for normalizer in self.model.normalizers:
            normalizer.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
            for normalizer in self.ema_model.normalizers:
                normalizer.to(device)
        optimizer_to(self.optimizer, device)
        
        # save batch for sampling
        train_sampling_batchs = []
        for i in range(cfg.task_num):
            train_sampling_batchs.append(None)
        # train_sampling_batch3 = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # ========================================
        def run_batch_with_timeout(loader_iter, timeout_sec=10):
            result = {'batch': None, 'error': None}

            def fetch():
                try:
                    result['batch'] = next(loader_iter)
                except Exception as e:
                    result['error'] = e

            t = threading.Thread(target=fetch)
            t.start()
            t.join(timeout_sec)
            if t.is_alive():
                print(f"[Timeout] Batch load exceeded {timeout_sec}s, skipping.")
                return None
            if result['error']:
                print(f"[Error] Exception during batch fetch: {result['error']}")
                return None
            return result['batch']
        # ========================================


        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()

                with tqdm.tqdm(multi_traindataloader, desc=f"Training epoch {self.epoch}",
                               leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:

                    loader_iter = iter(tepoch)

                    for batch_idx in range(max_train_dataloader_len):
                        batch = run_batch_with_timeout(loader_iter, timeout_sec=10)
                        if batch is None:
                            print(f"[Batch {batch_idx}] skipped due to timeout or error.")
                            continue  # skip this step, don't run training

                        assigned_task_id = batch_idx % cfg.task_num
                        assert assigned_task_id == multi_traindataloader.loader_idx

                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        task_id = torch.tensor([assigned_task_id], dtype=torch.int64).to(device)
                        if train_sampling_batchs[assigned_task_id] is None:
                            print("Assigning train_sampling_batch with task_id: ", assigned_task_id)
                            train_sampling_batchs[assigned_task_id] = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch, task_id)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (max_train_dataloader_len - 1))
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                                and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                for i, train_sampling_batch in enumerate(train_sampling_batchs):
                    if train_sampling_batch is None:
                        raise ValueError(f"train_sampling_batch {i} is None")
                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses_list = []
                        for i in range(cfg.task_num):
                            val_losses_list.append([])
                        zip_val_dataloaders = zip_longest(*val_dataloaders)
                        # val_losses3 = list()
                        with tqdm.tqdm(zip_val_dataloaders, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batches in enumerate(tepoch):
                                for i, batch in enumerate(batches):
                                    if batch is None:
                                        continue
                                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                    loss = self.model.compute_loss(batch,task_id=torch.tensor([i], dtype=torch.int64).to(device))
                                    val_losses_list[i].append(loss)
                                    if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps-1):
                                        break
                        if len(val_losses_list[0]) > 0:
                            for i, val_losses in enumerate(val_losses_list):
                                val_loss = torch.mean(torch.tensor(val_losses)).item()
                                step_log[f'val_loss_{i}'] = val_loss
                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        for i, train_sampling_batch in enumerate(train_sampling_batchs):
                            assert train_sampling_batch is not None
                            batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            obs_dict = batch['obs']
                            gt_action = batch['action']
                            result = policy.predict_action(obs_dict,task_id=torch.tensor([i], dtype=torch.int64).to(device))
                            pred_action = result['action_pred'] 
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                            step_log[f'train_action_mse_error_{i}'] = mse.item()
                            del batch
                            del obs_dict
                            del gt_action
                            del result
                            del pred_action
                            del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    sum=0
                    for key in metric_dict.keys():
                        # if start with cfg.checkpoint.topk.monitor_key, then sum up
                        if key.startswith(cfg.checkpoint.topk.monitor_key):
                            sum+=metric_dict[key]
                    metric_dict[cfg.checkpoint.topk.monitor_key] = sum
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
                multi_traindataloader.reset()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainMetadriveDiffusionTransformerWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
