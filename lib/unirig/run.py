import argparse
import yaml
from box import Box
import os
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import WandbLogger
from typing import List
from math import ceil
import numpy as np
from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy
from src.inference.download import download

from src.data.asset import Asset
from src.data.extract import get_files
from src.data.dataset import UniRigDatasetModule, DatasetConfig, ModelInput
from src.data.datapath import Datapath
from src.data.transform import TransformConfig
from src.tokenizer.spec import TokenizerConfig
from src.tokenizer.parse import get_tokenizer
from src.model.parse import get_model
from src.system.parse import get_system, get_writer

from tqdm import tqdm
import time
import json

def load(task: str, path: str) -> Box:
    if path.endswith('.yaml'):
        path = path.removesuffix('.yaml')
    path += '.yaml'
    print(f"\033[92mload {task} config: {path}\033[0m")
    return Box(yaml.safe_load(open(path, 'r')))

def apply_config_overrides(config, overrides: dict):
    """
    Apply config overrides to a Box/dict config object.

    Args:
        config: Box or dict config object
        overrides: Dictionary of override values

    Returns:
        Modified config object
    """
    if not overrides:
        return config

    # Apply overrides to transform config (sampler and vertex group)
    if hasattr(config, 'predict_transform_config') or 'predict_transform_config' in config:
        ptc = config.get('predict_transform_config', {})

        # Override sampler config
        if 'sampler_config' in ptc:
            if 'num_samples' in overrides:
                ptc['sampler_config']['num_samples'] = overrides['num_samples']
            if 'vertex_samples' in overrides:
                ptc['sampler_config']['vertex_samples'] = overrides['vertex_samples']

        # Override vertex group config (voxel_skin)
        if 'vertex_group_config' in ptc and 'kwargs' in ptc['vertex_group_config']:
            vg_kwargs = ptc['vertex_group_config']['kwargs']
            if 'voxel_skin' in vg_kwargs:
                vs = vg_kwargs['voxel_skin']
                if 'voxel_grid_size' in overrides:
                    vs['grid'] = overrides['voxel_grid_size']
                # Map voxel_mask_power to alpha (voxel_mask_power is the UI name)
                if 'voxel_mask_power' in overrides:
                    vs['alpha'] = overrides['voxel_mask_power']
                elif 'alpha' in overrides:
                    vs['alpha'] = overrides['alpha']
                if 'grid_query' in overrides:
                    vs['grid_query'] = overrides['grid_query']
                if 'vertex_query' in overrides:
                    vs['vertex_query'] = overrides['vertex_query']
                if 'grid_weight' in overrides:
                    vs['grid_weight'] = overrides['grid_weight']

    return config

def nullable_string(val):
    if not val:
        return None
    return val

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, required=False, default=123,
                        help="random seed")
    parser.add_argument("--input", type=nullable_string, required=False, default=None,
                        help="a single input file or files splited by comma")
    parser.add_argument("--input_dir", type=nullable_string, required=False, default=None,
                        help="input directory")
    parser.add_argument("--output", type=nullable_string, required=False, default=None,
                        help="filename for a single output")
    parser.add_argument("--output_dir", type=nullable_string, required=False, default=None,
                        help="output directory")
    parser.add_argument("--npz_dir", type=nullable_string, required=False, default='tmp',
                        help="intermediate npz directory")
    parser.add_argument("--cls", type=nullable_string, required=False, default=None,
                        help="class name")
    parser.add_argument("--data_name", type=nullable_string, required=False, default=None,
                        help="npz filename from skeleton phase")
    args = parser.parse_args()
    
    L.seed_everything(args.seed, workers=True)
    
    task = load('task', args.task)
    mode = task.mode
    assert mode in ['train', 'predict', 'validate']
    
    if args.input is not None or args.input_dir is not None:
        assert args.output_dir is not None or args.output is not None, 'output or output_dir must be specified'
        assert args.npz_dir is not None, 'npz_dir must be specified'
        files = get_files(
            data_name=task.components.data_name,
            inputs=args.input,
            input_dataset_dir=args.input_dir,
            output_dataset_dir=args.npz_dir,
            force_override=True,
            warning=False,
        )
        files = [f[1] for f in files]
        if len(files) > 1 and args.output is not None:
            print("\033[92mwarning: output is specified, but multiple files are detected. Output will be written.\033[0m")
        datapath = Datapath(files=files, cls=args.cls)
    else:
        datapath = None
    
    data_config = load('data', os.path.join('configs/data', task.components.data))
    transform_config = load('transform', os.path.join('configs/transform', task.components.transform))

    # Check for config overrides from environment variable
    config_overrides_json = os.environ.get('UNIRIG_CONFIG_OVERRIDES', None)
    if config_overrides_json:
        try:
            config_overrides = json.loads(config_overrides_json)
            print(f"\033[92mApplying config overrides: {config_overrides}\033[0m")
            transform_config = apply_config_overrides(transform_config, config_overrides)
        except json.JSONDecodeError as e:
            print(f"\033[91mWarning: Failed to parse config overrides: {e}\033[0m")

    # get tokenizer
    tokenizer_config = task.components.get('tokenizer', None)
    if tokenizer_config is not None:
        tokenizer_config = load('tokenizer', os.path.join('configs/tokenizer', task.components.tokenizer))
        tokenizer_config = TokenizerConfig.parse(config=tokenizer_config)
    
    # get data name
    data_name = task.components.get('data_name', 'raw_data.npz')
    if args.data_name is not None:
        data_name = args.data_name
        
    # get train dataset
    train_dataset_config = data_config.get('train_dataset_config', None)
    if train_dataset_config is not None:
        train_dataset_config = DatasetConfig.parse(config=train_dataset_config)
    
    # get train transform
    train_transform_config = transform_config.get('train_transform_config', None)
    if train_transform_config is not None:
        train_transform_config = TransformConfig.parse(config=train_transform_config)
        
    # get predict dataset
    predict_dataset_config = data_config.get('predict_dataset_config', None)
    if predict_dataset_config is not None:
        predict_dataset_config = DatasetConfig.parse(config=predict_dataset_config).split_by_cls()
    
    # get predict transform
    predict_transform_config = transform_config.get('predict_transform_config', None)
    if predict_transform_config is not None:
        predict_transform_config = TransformConfig.parse(config=predict_transform_config)
        
    # get validate dataset
    validate_dataset_config = data_config.get('validate_dataset_config', None)
    if validate_dataset_config is not None:
        validate_dataset_config = DatasetConfig.parse(config=validate_dataset_config).split_by_cls()
    
    # get validate transform
    validate_transform_config = transform_config.get('validate_transform_config', None)
    if validate_transform_config is not None:
        validate_transform_config = TransformConfig.parse(config=validate_transform_config)
    
    # get model
    model_config = task.components.get('model', None)
    if model_config is not None:
        model_config = load('model', os.path.join('configs/model', model_config))
        if tokenizer_config is not None:
            tokenizer = get_tokenizer(config=tokenizer_config)
        else:
            tokenizer = None
        model = get_model(tokenizer=tokenizer, **model_config)
    else:
        model = None
    
    # set data
    data = UniRigDatasetModule(
        process_fn=None if model is None else model._process_fn,
        train_dataset_config=train_dataset_config,
        predict_dataset_config=predict_dataset_config,
        predict_transform_config=predict_transform_config,
        validate_dataset_config=validate_dataset_config,
        train_transform_config=train_transform_config,
        validate_transform_config=validate_transform_config,
        tokenizer_config=tokenizer_config,
        debug=False,
        data_name=data_name,
        datapath=datapath,
        cls=args.cls,
    )
    
    # add call backs
    callbacks = []

    ## get checkpoint callback
    checkpoint_config = task.get('checkpoint', None)
    if checkpoint_config is not None:
        checkpoint_config['dirpath'] = os.path.join('experiments', task.experiment_name)
        callbacks.append(ModelCheckpoint(**checkpoint_config))
    
    ## get writer callback
    writer_config = task.get('writer', None)
    if writer_config is not None:
        assert predict_transform_config is not None, 'missing predict_transform_config in transform'
        if args.output_dir is not None or args.output is not None:
            if args.output is not None:
                assert args.output.endswith('.fbx'), 'output must be .fbx'
            writer_config['npz_dir'] = args.npz_dir
            writer_config['output_dir'] = args.output_dir
            writer_config['output_name'] = args.output
            writer_config['user_mode'] = True
        callbacks.append(get_writer(**writer_config, order_config=predict_transform_config.order_config))
    
    # get trainer
    trainer_config = task.get('trainer', {})
    
    # get scheduler
    scheduler_config = task.get('scheduler', None)
    
    optimizer_config = task.get('optimizer', None)
    loss_config = task.get('loss', None)
    
    # get system
    system_config = task.components.get('system', None)
    if system_config is not None:
        system_config = load('system', os.path.join('configs/system', system_config))
        system = get_system(
            **system_config,
            model=model,
            optimizer_config=optimizer_config,
            loss_config=loss_config,
            scheduler_config=scheduler_config,
            steps_per_epoch=1 if train_dataset_config is None else 
            ceil(len(data.train_dataloader()) // trainer_config.devices // trainer_config.num_nodes),
        )
    else:
        system = None
    
    wandb_config = task.get('wandb', None)
    if wandb_config is not None:
        logger = WandbLogger(
            config={
                'task': task,
                'data': data_config,
                'tokenizer': tokenizer_config,
                'train_dataset_config': train_dataset_config,
                'validate_dataset_config': validate_dataset_config,
                'predict_dataset_config': predict_dataset_config,
                'train_transform_config': train_transform_config,
                'validate_transform_config': validate_transform_config,
                'predict_transform_config': predict_transform_config,
                'model_config': model_config,
                'optimizer_config': optimizer_config,
                'system_config': system_config,
                'checkpoint_config': checkpoint_config,
                'writer_config': writer_config,
            },
            log_model=True,
            **wandb_config
        )
        if logger.experiment.id is not None:
            print(f"\033[92mWandbLogger started: {logger.experiment.id}\033[0m")
            # Get the run URL using wandb.run.get_url() which is more reliable
            run_url = logger.experiment.get_url() if hasattr(logger.experiment, 'get_url') else logger.experiment.url
            print(f"\033[92mWandbLogger url: {run_url}\033[0m")
        else:
            print("\033[91mWandbLogger failed to start\033[0m")
    else:
        logger = None

    # set ckpt path
    resume_from_checkpoint = task.get('resume_from_checkpoint', None)
    resume_from_checkpoint = download(resume_from_checkpoint)
    if trainer_config.get('strategy', None) == "fsdp":
        strategy = FSDPStrategy(
            # Enable activation checkpointing on these layers
            auto_wrap_policy={
                torch.nn.MultiheadAttention
            },
            activation_checkpointing_policy={
                torch.nn.Linear,
                torch.nn.MultiheadAttention,
            },
        )
        trainer_config['strategy'] = strategy
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=logger,
        **trainer_config,
    )
    
    if mode == 'train':
        trainer.fit(system, datamodule=data, ckpt_path=resume_from_checkpoint)
    elif mode == 'predict':
        assert resume_from_checkpoint is not None, 'expect resume_from_checkpoint in task'
        trainer.predict(system, datamodule=data, ckpt_path=resume_from_checkpoint, return_predictions=False, weights_only=False)
    elif mode == 'validate':
        trainer.validate(system, datamodule=data, ckpt_path=resume_from_checkpoint)
    else:
        assert 0