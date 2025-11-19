"""
Subprocess script to run UniRig skinning inference.
This runs in a fresh Python process for better quality.
"""

import os
import sys
import argparse
from pathlib import Path

# Add UniRig to path
LIB_DIR = Path(__file__).parent.resolve()
UNIRIG_PATH = LIB_DIR / "unirig"
sys.path.insert(0, str(UNIRIG_PATH))

import torch
import lightning as L
from src.inference.download import download
from src.data.extract import get_files
from src.data.datapath import Datapath
from src.data.dataset import UniRigDatasetModule, DatasetConfig
from src.data.transform import TransformConfig
from src.system.skin import SkinSystem
from src.model.parse import get_model
from src.tokenizer.parse import get_tokenizer
from src.system.parse import get_system, get_writer
import yaml
from box import Box


def load_yaml_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return Box(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input GLB file')
    parser.add_argument('--output', required=True, help='Output FBX file')
    parser.add_argument('--npz_dir', required=True, help='Directory containing NPZ files')
    parser.add_argument('--data_name', default='predict_skeleton.npz', help='NPZ filename')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint name')
    parser.add_argument('--voxel_grid_size', type=int, default=196)
    parser.add_argument('--num_samples', type=int, default=32768)
    parser.add_argument('--vertex_samples', type=int, default=8192)
    parser.add_argument('--voxel_mask_power', type=float, default=0.5)
    args = parser.parse_args()

    # Change to UniRig directory
    os.chdir(str(UNIRIG_PATH))

    # Set seed
    L.seed_everything(args.seed, workers=True)

    print(f"[Subprocess Skinning] Loading model...")
    print(f"[Subprocess Skinning] Checkpoint: {args.checkpoint}")
    print(f"[Subprocess Skinning] Config: voxel_grid={args.voxel_grid_size}, num_samples={args.num_samples}")

    # Load task config
    task_config_path = os.path.join(str(UNIRIG_PATH), 'configs/task', 'quick_inference_unirig_skin.yaml')
    task = load_yaml_config(task_config_path)

    # Apply config overrides
    if hasattr(task, 'predict_transform_config'):
        ptc = task.predict_transform_config
        if 'sampler_config' in ptc:
            ptc.sampler_config.num_samples = args.num_samples
            ptc.sampler_config.vertex_samples = args.vertex_samples
        if 'vertex_group_config' in ptc and 'kwargs' in ptc.vertex_group_config:
            vg_kwargs = ptc.vertex_group_config.kwargs
            if 'voxel_skin' in vg_kwargs:
                vg_kwargs.voxel_skin.grid = args.voxel_grid_size
                vg_kwargs.voxel_skin.alpha = args.voxel_mask_power

    # Load tokenizer config
    tokenizer_config_name = task.components.get('tokenizer', None)
    if tokenizer_config_name:
        # Add .yaml extension if not present
        if not tokenizer_config_name.endswith('.yaml'):
            tokenizer_config_name = f"{tokenizer_config_name}.yaml"
        tokenizer_config = load_yaml_config(os.path.join(str(UNIRIG_PATH), 'configs/tokenizer', tokenizer_config_name))
        tokenizer = get_tokenizer(config=tokenizer_config)
    else:
        tokenizer_config = None
        tokenizer = None

    # Build model
    model_config_name = task.components.get('model', None)
    if model_config_name:
        # Add .yaml extension if not present
        if not model_config_name.endswith('.yaml'):
            model_config_name = f"{model_config_name}.yaml"
        model_config = load_yaml_config(os.path.join(str(UNIRIG_PATH), 'configs/model', model_config_name))
        model = get_model(tokenizer=tokenizer, **model_config)
    else:
        model = None

    # Download and load checkpoint
    checkpoint_path = download(args.checkpoint)

    # Load system
    system_config_name = task.components.get('system', None)
    if system_config_name:
        # Add .yaml extension if not present
        if not system_config_name.endswith('.yaml'):
            system_config_name = f"{system_config_name}.yaml"
        system_config = load_yaml_config(os.path.join(str(UNIRIG_PATH), 'configs/system', system_config_name))
        system = get_system(
            **system_config,
            model=model,
            optimizer_config=None,
            loss_config=None,
            scheduler_config=None,
            steps_per_epoch=1,
        )
    else:
        raise RuntimeError("No system config found")

    # Load checkpoint weights
    print(f"[Subprocess Skinning] Loading checkpoint weights...")
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if torch_version >= (2, 6):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    system.load_state_dict(checkpoint['state_dict'], strict=False)

    if torch.cuda.is_available():
        system = system.cuda()
        print(f"[Subprocess Skinning] Model on GPU")
    else:
        print(f"[Subprocess Skinning] Model on CPU")

    system.eval()

    # Prepare data
    print(f"[Subprocess Skinning] Preparing data...")
    files = get_files(
        data_name=task.components.data_name,
        inputs=args.input,
        input_dataset_dir=None,
        output_dataset_dir=args.npz_dir,
        force_override=True,
        warning=False,
    )
    files = [f[1] for f in files]
    datapath = Datapath(files=files, cls=None)

    # Load data and transform configs (add .yaml extension if needed)
    data_config_name = task.components.data
    if not data_config_name.endswith('.yaml'):
        data_config_name = f"{data_config_name}.yaml"
    data_config = load_yaml_config(os.path.join(str(UNIRIG_PATH), 'configs/data', data_config_name))

    transform_config_name = task.components.transform
    if not transform_config_name.endswith('.yaml'):
        transform_config_name = f"{transform_config_name}.yaml"
    transform_config = load_yaml_config(os.path.join(str(UNIRIG_PATH), 'configs/transform', transform_config_name))

    # Get data name
    data_name_actual = task.components.get('data_name', 'raw_data.npz')
    if args.data_name:
        data_name_actual = args.data_name

    # Get predict dataset config
    predict_dataset_config = data_config.get('predict_dataset_config', None)
    if predict_dataset_config is not None:
        predict_dataset_config = DatasetConfig.parse(config=predict_dataset_config).split_by_cls()

    # Get predict transform config
    predict_transform_config = transform_config.get('predict_transform_config', None)
    if predict_transform_config is not None:
        predict_transform_config = TransformConfig.parse(config=predict_transform_config)

    # Create data module
    data = UniRigDatasetModule(
        process_fn=system.model._process_fn if system and system.model else None,
        train_dataset_config=None,
        predict_dataset_config=predict_dataset_config,
        predict_transform_config=predict_transform_config,
        validate_dataset_config=None,
        train_transform_config=None,
        validate_transform_config=None,
        tokenizer_config=tokenizer_config,
        debug=False,
        data_name=data_name_actual,
        datapath=datapath,
        cls=None,
    )

    # Setup callbacks with writer
    callbacks = []
    writer_config = task.get('writer', None)
    if writer_config is not None:
        writer_config['npz_dir'] = args.npz_dir
        writer_config['output_dir'] = None
        writer_config['output_name'] = args.output
        writer_config['user_mode'] = True
        callbacks.append(get_writer(**writer_config, order_config=predict_transform_config.order_config))

    # Run inference
    print(f"[Subprocess Skinning] Running inference...")
    trainer = L.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=False,
        enable_checkpointing=False,
        callbacks=callbacks,
    )

    predictions = trainer.predict(system, data)

    print(f"[Subprocess Skinning] Inference complete!")
    print(f"[Subprocess Skinning] Output written to: {args.output}")


if __name__ == '__main__':
    main()
