import os
os.environ["WANDB_MODE"] = "offline"

import datetime
import logging
import os
import random
import re
import shutil
from typing import List, Optional

import braceexpand
import fire
import torch
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusionXLPipeline
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torchvision.transforms import InterpolationMode

from lbm.data.datasets import DataModule, DataModuleConfig
from lbm.data.filters import KeyFilter, KeyFilterConfig
from lbm.data.mappers import (
    KeyRenameMapper,
    KeyRenameMapperConfig,
    MapperWrapper,
    RescaleMapper,
    RescaleMapperConfig,
    TorchvisionMapper,
    TorchvisionMapperConfig,
)
from lbm.models.embedders import (
    ConditionerWrapper,
    LatentsConcatEmbedder,
    LatentsConcatEmbedderConfig,
)
from lbm.models.lbm import LBMConfig, LBMModel
from lbm.models.unets import DiffusersUNet2DCondWrapper
from lbm.models.vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig
from lbm.trainer import TrainingConfig, TrainingPipeline
from lbm.trainer.loggers import WandbSampleLogger
from lbm.trainer.utils import StateDictAdapter

from torch.utils.data import DataLoader  # 新增
from pytorch_lightning import LightningDataModule  # 新增
from de_relight_dataset import RelightPairsDataset, relight_collate  # 新增（我们前面写好的 dataset 文件）
# from olat_white_dataset import RelightPairsDataset, relight_collate  # 新增（我们前面写好的 dataset 文件）

import os
import torch
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid, save_image

class LocalImageLogger(Callback):
    def __init__(self, every_n_steps=500, out_dir="train_images"):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step
        if global_step % self.every_n_steps != 0:
            return

        pl_module.eval()
        with torch.no_grad():
            logs = pl_module.log_samples(batch)

        if logs is None:
            return

        # 遍历 logs 里的每个 key
        for key, value in logs.items():
            if isinstance(value, torch.Tensor) and value.dim() == 4:  # [B, C, H, W]
                # grid = make_grid(value[:4], nrow=4, normalize=True, value_range=(-1, 1))
                grid = make_grid(value[:4], nrow=4, normalize=True, value_range=(0, 1))
                save_path = os.path.join(self.out_dir, f"step_{global_step:06d}_{key}.png")
                save_image(grid, save_path)
                print(f"[LocalImageLogger] Saved {key} → {save_path}")

        pl_module.train()



def get_model(
    backbone_signature: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_num_channels: int = 4,
    unet_input_channels: int = 4,
    timestep_sampling: str = "log_normal",
    selected_timesteps: Optional[List[float]] = None,
    prob: Optional[List[float]] = None,
    conditioning_images_keys: Optional[List[str]] = [],
    conditioning_masks_keys: Optional[List[str]] = [],
    source_key: str = "source_image",
    target_key: str = "source_image_paste",
    mask_key: str = "mask",
    bridge_noise_sigma: float = 0.0,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    pixel_loss_type: str = "lpips",
    latent_loss_type: str = "l2",
    latent_loss_weight: float = 1.0,
    pixel_loss_weight: float = 0.0,
):

    conditioners = []

    # Load pretrained model as base
    pipe = StableDiffusionXLPipeline.from_pretrained(
        backbone_signature,
        torch_dtype=torch.bfloat16,
    )

    ### MMMDiT ###
    # Get Architecture
    denoiser = DiffusersUNet2DCondWrapper(
        in_channels=unet_input_channels,  # Add downsampled_image
        out_channels=vae_num_channels,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=[
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ],
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
        only_cross_attention=False,
        block_out_channels=[320, 640, 1280],
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        dropout=0.0,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=[320, 640, 1280],
        transformer_layers_per_block=[1, 2, 10],
        reverse_transformer_layers_per_block=None,
        encoder_hid_dim=None,
        encoder_hid_dim_type=None,
        attention_head_dim=[5, 10, 20],
        num_attention_heads=None,
        dual_cross_attention=False,
        use_linear_projection=True,
        class_embed_type=None,
        addition_embed_type=None,
        addition_time_embed_dim=None,
        num_class_embeds=None,
        upcast_attention=None,
        resnet_time_scale_shift="default",
        resnet_skip_time_act=False,
        resnet_out_scale_factor=1.0,
        time_embedding_type="positional",
        time_embedding_dim=None,
        time_embedding_act_fn=None,
        timestep_post_act=None,
        time_cond_proj_dim=None,
        conv_in_kernel=3,
        conv_out_kernel=3,
        projection_class_embeddings_input_dim=None,
        attention_type="default",
        class_embeddings_concat=False,
        mid_block_only_cross_attention=None,
        cross_attention_norm=None,
        addition_embed_type_num_heads=64,
    ).to(torch.bfloat16)

    state_dict = pipe.unet.state_dict()

    del state_dict["add_embedding.linear_1.weight"]
    del state_dict["add_embedding.linear_1.bias"]
    del state_dict["add_embedding.linear_2.weight"]
    del state_dict["add_embedding.linear_2.bias"]

    # Adapt the shapes
    state_dict_adapter = StateDictAdapter()
    state_dict = state_dict_adapter(
        model_state_dict=denoiser.state_dict(),
        checkpoint_state_dict=state_dict,
        regex_keys=[
            r"class_embedding.linear_\d+.(weight|bias)",
            r"conv_in.weight",
            r"(down_blocks|up_blocks)\.\d+\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
            r"mid_block\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
        ],
        strategy="zeros",
    )

    denoiser.load_state_dict(state_dict, strict=True)

    del pipe

    if conditioning_images_keys != [] or conditioning_masks_keys != []:

        latents_concat_embedder_config = LatentsConcatEmbedderConfig(
            image_keys=conditioning_images_keys,
            mask_keys=conditioning_masks_keys,
        )
        latent_concat_embedder = LatentsConcatEmbedder(latents_concat_embedder_config)
        latent_concat_embedder.freeze()
        conditioners.append(latent_concat_embedder)

    # Wrap conditioners and set to device
    conditioner = ConditionerWrapper(
        conditioners=conditioners,
    )

    ## VAE ##
    # Get VAE model
    vae_config = AutoencoderKLDiffusersConfig(
        version=backbone_signature,
        subfolder="vae",
        tiling_size=(128, 128),
    )
    vae = AutoencoderKLDiffusers(vae_config)
    vae.freeze()
    vae.to(torch.bfloat16)

    # LBM Config
    config = LBMConfig(
        ucg_keys=None,
        source_key=source_key,
        target_key=target_key,
        mask_key=mask_key,
        latent_loss_weight=latent_loss_weight,
        latent_loss_type=latent_loss_type,
        pixel_loss_type=pixel_loss_type,
        pixel_loss_weight=pixel_loss_weight,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        selected_timesteps=selected_timesteps,
        prob=prob,
        bridge_noise_sigma=bridge_noise_sigma,
    )

    training_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler",
    )
    sampling_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler",
    )

    # LBM Model
    model = LBMModel(
        config,
        denoiser=denoiser,
        training_noise_scheduler=training_noise_scheduler,
        sampling_noise_scheduler=sampling_noise_scheduler,
        vae=vae,
        conditioner=conditioner,
    ).to(torch.bfloat16)

    return model


def get_filter_mappers():
    filters_mappers = [
        KeyFilter(KeyFilterConfig(keys=["jpg", "normal_aligned.png", "mask.png"])),
        MapperWrapper(
            [
                KeyRenameMapper(
                    KeyRenameMapperConfig(
                        key_map={
                            "jpg": "image",
                            "normal_aligned.png": "normal",
                            "mask.png": "mask",
                        }
                    )
                ),
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="image",
                        transforms=["ToTensor", "Resize"],
                        transforms_kwargs=[
                            {},
                            {
                                "size": (480, 640),
                                "interpolation": InterpolationMode.NEAREST_EXACT,
                            },
                        ],
                    )
                ),
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="normal",
                        transforms=["ToTensor", "Resize"],
                        transforms_kwargs=[
                            {},
                            {
                                "size": (480, 640),
                                "interpolation": InterpolationMode.NEAREST_EXACT,
                            },
                        ],
                    )
                ),
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="mask",
                        transforms=["ToTensor", "Resize", "Normalize"],
                        transforms_kwargs=[
                            {},
                            {
                                "size": (480, 640),
                                "interpolation": InterpolationMode.NEAREST_EXACT,
                            },
                            {"mean": 0.0, "std": 1.0},
                        ],
                    )
                ),
                RescaleMapper(RescaleMapperConfig(key="image")),
                RescaleMapper(RescaleMapperConfig(key="normal")),
            ],
        ),
    ]

    return filters_mappers

# ========= 新增：FolderRelightDataModule 适配器（最小侵入，替代 webdataset） =========
class FolderRelightDataModule(LightningDataModule):
    def __init__(
        self,
        composite_root: str,
        hdr_root: str,
        alpha_root: str,
        batch_size: int = 8,
        num_workers: int = 8,
        mode: str = "random_per_epoch",   # 或 "all_pairs"
        hdr_out_hw=(64, 128),
        resize_hw=None,
    ):
        super().__init__()
        self.composite_root = composite_root
        self.hdr_root = hdr_root
        self.alpha_root = alpha_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.hdr_out_hw = hdr_out_hw
        self.resize_hw = resize_hw
        self.train_ds = None
        self.val_ds = None  # 如需验证集可后续补

    def setup(self, stage=None):
        # 训练集
        self.train_ds = RelightPairsDataset(
            composite_root=self.composite_root,
            hdr_root=self.hdr_root,
            alpha_root=self.alpha_root,
            mode=self.mode,
            hdr_out_hw=self.hdr_out_hw,
            resize_hw=self.resize_hw,
        )
        # 目前没有单独验证集 —— 返回 None 即可不影响训练流程
        self.val_ds = None

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=relight_collate,
        )

    def val_dataloader(self):
        # 如未来需要验证集，这里返回 DataLoader(self.val_ds, ...)
        return None


# ========= 替换：get_data_module 增加 folder 分支 =========
def get_data_module(
    train_shards: List[str],
    validation_shards: List[str],
    batch_size: int,
    # 新增三个可选参数（不破坏原签名）
    composite_root: Optional[str] = None,
    hdr_root: Optional[str] = None,
    alpha_root: Optional[str] = None,
    # 可选：给 folder 分支传入的 worker 数等
    num_workers: int = 8,
):
    # --- 如果提供了三条路径：走 folder dataset（我们的配对/HDR/mask 规则） ---
    if composite_root and hdr_root and alpha_root:
        data_module = FolderRelightDataModule(
            composite_root=composite_root,
            hdr_root=hdr_root,
            alpha_root=alpha_root,
            batch_size=batch_size,
            num_workers=num_workers,
            mode="random_per_epoch",  # 也可换 "all_pairs"
            hdr_out_hw=(1024, 750),
            resize_hw=None,
        )
        return data_module

    # --- 否则：保留原有 webdataset 分支（原实现保持不动） ---
    train_filters_mappers = get_filter_mappers()

    # unbrace urls
    train_shards_path_or_urls_unbraced = []
    for train_shards_path_or_url in train_shards:
        train_shards_path_or_urls_unbraced.extend(list(braceexpand.braceexpand(train_shards_path_or_url)))

    validation_shards_path_or_urls_unbraced = []
    for validation_shards_path_or_url in validation_shards:
        validation_shards_path_or_urls_unbraced.extend(list(braceexpand.braceexpand(validation_shards_path_or_url)))

    train_data_config = DataModuleConfig(
        shards_path_or_urls=train_shards_path_or_urls_unbraced,
        decoder="pil",
        shuffle_before_split_by_node_buffer_size=10,
        shuffle_before_split_by_workers_buffer_size=10,
        shuffle_before_filter_mappers_buffer_size=10,
        shuffle_after_filter_mappers_buffer_size=10,
        per_worker_batch_size=batch_size,
        num_workers=min(10, len(train_shards_path_or_urls_unbraced)),
    )

    validation_filters_mappers = get_filter_mappers()

    data_config = DataModuleConfig(
        shards_path_or_urls=validation_shards_path_or_urls_unbraced,
        decoder="pil",
        shuffle_before_split_by_node_buffer_size=10,
        shuffle_before_split_by_workers_buffer_size=10,
        shuffle_before_filter_mappers_buffer_size=10,
        shuffle_after_filter_mappers_buffer_size=10,
        per_worker_batch_size=batch_size,
        num_workers=min(10, len(train_shards_path_or_urls_unbraced)),
    )

    validation_data_config = data_config

    data_module = DataModule(
        train_config=train_data_config,
        train_filters_mappers=train_filters_mappers,
        eval_config=validation_data_config,
        eval_filters_mappers=validation_filters_mappers,
    )
    return data_module



def main(
    train_shards: List[str] = ["pipe:cat path/to/train/shards"],
    validation_shards: List[str] = ["pipe:cat path/to/validation/shards"],
    backbone_signature: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_num_channels: int = 4,
    unet_input_channels: int = 4,
    source_key: str = "image",
    target_key: str = "target",
    mask_key: str = "mask",
    composite_root: Optional[str] = None,
    hdr_root: Optional[str] = None,
    alpha_root: Optional[str] = None,
    wandb_project: str = "lbm-surface",
    batch_size: int = 8,
    num_steps: List[int] = [1, 2, 4],
    learning_rate: float = 5e-5,
    learning_rate_scheduler: str = None,
    learning_rate_scheduler_kwargs: dict = {},
    optimizer: str = "AdamW",
    optimizer_kwargs: dict = {},
    timestep_sampling: str = "uniform",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    pixel_loss_type: str = "lpips",
    latent_loss_type: str = "l2",
    latent_loss_weight: float = 1.0,
    pixel_loss_weight: float = 0.0,
    selected_timesteps: List[float] = None,
    prob: List[float] = None,
    conditioning_images_keys: Optional[List[str]] = [],
    conditioning_masks_keys: Optional[List[str]] = [],
    config_yaml: dict = None,
    save_ckpt_path: str = "./checkpoints_relight_1008",
    log_interval: int = 100,
    resume_from_checkpoint: bool = True,
    max_epochs: int = 100,
    bridge_noise_sigma: float = 0.005,
    save_interval: int = 1000,
    path_config: str = None,
):
    model = get_model(
        backbone_signature=backbone_signature,
        vae_num_channels=vae_num_channels,
        unet_input_channels=unet_input_channels,
        source_key="image",
        target_key="target",
        mask_key="mask",
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        pixel_loss_type=pixel_loss_type,
        latent_loss_type=latent_loss_type,
        latent_loss_weight=latent_loss_weight,
        pixel_loss_weight=pixel_loss_weight,
        selected_timesteps=selected_timesteps,
        prob=prob,
        conditioning_images_keys=["hdr_env"],
        conditioning_masks_keys=["mask"],
        bridge_noise_sigma=bridge_noise_sigma,
    )

    data_module = get_data_module(
        train_shards=train_shards,
        validation_shards=validation_shards,
        batch_size=batch_size,
        composite_root=composite_root,   # 新增
        hdr_root=hdr_root,               # 新增
        alpha_root=alpha_root,           # 新增
    )

    train_parameters = ["denoiser.*"]

    # Training Config
    training_config = TrainingConfig(
        learning_rate=learning_rate,
        lr_scheduler_name=learning_rate_scheduler,
        lr_scheduler_kwargs=learning_rate_scheduler_kwargs,
        log_keys=["image", "target", "mask", "uniform_fg"],
        trainable_params=train_parameters,
        optimizer_name=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        log_samples_model_kwargs={
            "input_shape": None,
            "num_steps": num_steps,
        },
    )
    if (
        os.path.exists(save_ckpt_path)
        and resume_from_checkpoint
        and "last.ckpt" in os.listdir(save_ckpt_path)
    ):
        start_ckpt = f"{save_ckpt_path}/last.ckpt"
        print(f"Resuming from checkpoint: {start_ckpt}")

    else:
        start_ckpt = None

    pipeline = TrainingPipeline(model=model, pipeline_config=training_config)

    pipeline.save_hyperparameters(
        {
            f"embedder_{i}": embedder.config.to_dict()
            for i, embedder in enumerate(model.conditioner.conditioners)
        }
    )

    pipeline.save_hyperparameters(
        {
            "denoiser": model.denoiser.config,
            "vae": model.vae.config.to_dict(),
            "config_yaml": config_yaml,
            "training": training_config.to_dict(),
            "training_noise_scheduler": model.training_noise_scheduler.config,
            "sampling_noise_scheduler": model.sampling_noise_scheduler.config,
        }
    )

    training_signature = (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "-LBM-Surface"
        + f"{os.environ['SLURM_JOB_ID']}"
        + f"_{os.environ.get('SLURM_ARRAY_TASK_ID', 0)}"
    )
    dir_path = f"{save_ckpt_path}/logs/{training_signature}"
    if os.environ["SLURM_PROCID"] == "0":
        os.makedirs(dir_path, exist_ok=True)
        if path_config is not None:
            shutil.copy(path_config, f"{save_ckpt_path}/config.yaml")
    run_name = training_signature

    # Ignore parameters unused during training
    ignore_states = []
    for name, param in pipeline.model.named_parameters():
        ignore = True
        for regex in ["denoiser."]:
            pattern = re.compile(regex)
            if re.match(pattern, name):
                ignore = False
        if ignore:
            ignore_states.append(param)

    # FSDP Strategy
    strategy = FSDPStrategy(
        auto_wrap_policy=ModuleWrapPolicy(
            [
                UNet2DConditionModel,
                BasicTransformerBlock,
                ResnetBlock2D,
                torch.nn.Conv2d,
            ]
        ),
        activation_checkpointing_policy=ModuleWrapPolicy(
            [
                BasicTransformerBlock,
                ResnetBlock2D,
            ]
        ),
        sharding_strategy="SHARD_GRAD_OP",
        ignored_states=ignore_states,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=int(os.environ["SLURM_NPROCS"]) // int(os.environ["SLURM_NNODES"]),
        num_nodes=int(os.environ["SLURM_NNODES"]),
        strategy=strategy,
        default_root_dir="logs",
        logger=loggers.WandbLogger(
            project=wandb_project, offline=True, name=run_name, save_dir=save_ckpt_path
        ),
        callbacks=[
            WandbSampleLogger(log_batch_freq=log_interval),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=save_ckpt_path,
                every_n_train_steps=save_interval,
                save_last=True,
            ),
            LocalImageLogger(every_n_steps=200, out_dir="train_olat_1117_from_1105_final_48"),
        ],
        num_sanity_val_steps=0,
        precision="bf16-mixed",
        limit_val_batches=0,
        val_check_interval=1000,
        max_epochs=max_epochs,
    )

    trainer.fit(pipeline, data_module, ckpt_path=start_ckpt)


def main_from_config(path_config: str = None):
    with open(path_config, "r") as file:
        config = yaml.safe_load(file)
    logging.info(
        f"Running main with config: {yaml.dump(config, default_flow_style=False)}"
    )
    main(**config, config_yaml=config, path_config=path_config)


if __name__ == "__main__":
    fire.Fire(main_from_config)
    # fire.Fire(main)
