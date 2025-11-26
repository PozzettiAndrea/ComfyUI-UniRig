import os
import sys
import weakref
import torch
torch.multiprocessing.set_start_method('spawn')
import torch.nn as nn
import torch.utils.data
from functools import partial

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage
from pointcept.utils.registry import Registry

from sklearn.preprocessing import QuantileTransformer
from pointcept.utils.timer import Timer

TRAINERS = Registry("trainers")
from cuml.cluster.hdbscan import HDBSCAN
# from sklearn.cluster import HDBSCAN
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    o3d = None
    OPEN3D_AVAILABLE = False
import matplotlib.colors as mcolors
import numpy as np
from collections import OrderedDict
import trimesh
import pointops

class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter
        self._iter_timer = Timer()

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            # file_mode="a" if cfg.resume else "w",
            file_mode="a",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        # self.logger.info("=> Building val dataset & dataloader ...")
        # self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)

        # !!!
        self.model.scale_statistics = nn.Parameter(self.train_loader.scale_3d_statistics)
        self.model.scale_statistics.requires_grad = False
        self.model.quantile_transformer = self._get_quantile_func(self.train_loader.scale_3d_statistics)
        # print(id(self.model))
        # self.val_scales_list = [0.0, 0.5, 1.0, 1.5, 2.0]
        self.val_scales_list = self.cfg.val_scales_list
        self.mesh_voting = self.cfg.mesh_voting
        self.backbone_weight_path = self.cfg.backbone_weight_path

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            # self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            # data_time = self._iter_timer.seconds()
            # self.trainer.storage.put_scalar("data_time", data_time)
            # !!! load checkpoint
            if self.backbone_weight_path != None:
                self.logger.info("=> Loading checkpoint & weight ...")
                if os.path.isfile(self.backbone_weight_path):
                    checkpoint = torch.load(
                        self.backbone_weight_path,
                        map_location=lambda storage, loc: storage.cuda(),
                    )
                    weight = OrderedDict()
                    for key, value in checkpoint["state_dict"].items():
                        if not key.startswith("module."):
                            if comm.get_world_size() > 1:
                                key = "module." + key  # xxx.xxx -> module.xxx.xxx
                        # Now all keys contain "module." no matter DDP or not.
                        # if self.keywords in key:
                        #     key = key.replace(self.keywords, self.replacement)
                        if comm.get_world_size() == 1:
                            key = key[7:]  # module.xxx.xxx -> xxx.xxx
                        # if key.startswith("backbone."):
                        #     key = key[9:]  # backbone.xxx.xxx -> xxx.xxx
                        key = "backbone." + key  # xxx.xxx -> backbone.xxx.xxx
                        weight[key] = value
                    load_state_info = self.model.load_state_dict(weight, strict=False)
                    self.logger.info(f"Missing keys: {load_state_info[0]}")
                else:
                    self.logger.info(f"No weight found at: {self.backbone_weight_path}")

            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.model.train()
                loss_dict = self.model(self.train_loader.get_data(0))
                loss = loss_dict["instance_loss"]

                # !!! writer
                lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
                self.writer.add_scalar("lr", lr, self.epoch)
                for key in loss_dict.keys():
                    self.writer.add_scalar(
                        "train/" + key,
                        loss_dict[key].item(),
                        self.epoch,
                    )
                if self.epoch % 10 == 0:
                    self.logger.info(
                        f"iter: {self.epoch}, total_loss: {loss.item()}, loss_1: {loss_dict['instance_loss_1'].item()}, loss_2: {loss_dict['instance_loss_2'].item()}, loss_3: {loss_dict['instance_loss_3'].item()}"
                        )
                
                # !!! optimizer
                self.optimizer.zero_grad()
                if self.cfg.enable_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)

                    # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
                    # Fix torch warning scheduler step before optimizer step.
                    scaler = self.scaler.get_scale()
                    self.scaler.update()
                    if scaler <= self.scaler.get_scale():
                        self.scheduler.step()
                else:
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                # !!! save checkpoint
                if (self.epoch + 1) % 5000 == 0:
                    filename = os.path.join(self.cfg.save_path, "model", f"{str(self.epoch + 1)}.pth")
                    self.logger.info("Saving checkpoint to: " + filename)
                    torch.save(
                        {
                            "epoch": self.epoch + 1,
                            "state_dict": self.model.state_dict(),
                        },
                        filename + ".tmp",
                    )
                    os.replace(filename + ".tmp", filename)
            self.eval()
    
    def eval(self):
        # val_data = build_dataset(self.cfg.data.val)
        self.logger.info("=> Loading checkpoint & weight ...")
        if self.cfg.weight and os.path.isfile(self.cfg.weight):
            checkpoint = torch.load(
                self.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
            )
            load_state_info = self.model.load_state_dict(checkpoint["state_dict"])
            self.logger.info(f"Missing keys: {load_state_info[0]}")
        else:
            self.logger.info(f"No weight found at: {self.cfg.weight}")
            self.cfg.weight = "last"
    
        self.model.eval()
        save_root = os.path.join(self.cfg.save_path, "vis_pcd", os.path.splitext(os.path.basename(self.cfg.weight))[0])
        os.makedirs(save_root, exist_ok=True)
        group_save_root = os.path.join(self.cfg.save_path, "results", os.path.splitext(os.path.basename(self.cfg.weight))[0])
        os.makedirs(group_save_root, exist_ok=True)

        hex_colors = list(mcolors.CSS4_COLORS.values())
        rgb_colors = np.array([mcolors.to_rgb(color) for color in hex_colors if color not in ['#000000', '#FFFFFF']])
        def relative_luminance(color):
            return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
        rgb_colors = [color for color in rgb_colors if (relative_luminance(color) > 0.4 and relative_luminance(color) < 0.8)]
        np.random.shuffle(rgb_colors)
        input_dict = self.train_loader.val_data()

        pcd_inverse = self.train_loader.pcd_inverse
        if self.mesh_voting:
            mesh = trimesh.load(self.train_loader.mesh_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)

        for scale in self.val_scales_list:
            input_dict["scale"] = scale
            instance_feat = self.model(input_dict).cpu().detach().numpy()

            clusterer = HDBSCAN(
                cluster_selection_epsilon=0.1,
                min_samples=30,
                min_cluster_size=30,
                allow_single_cluster=False,
            ).fit(instance_feat)

            labels = clusterer.labels_
            invalid_label_mask = labels == -1
            if invalid_label_mask.sum() > 0:
                if invalid_label_mask.sum() == len(invalid_label_mask):
                    labels = np.zeros_like(labels)
                else:
                    coord = input_dict["obj"]["coord"].cuda().contiguous().float()
                    valid_coord = coord[~invalid_label_mask]
                    valid_offset = torch.tensor(valid_coord.shape[0]).cuda()
                    invalid_coord = coord[invalid_label_mask]
                    invalid_offset = torch.tensor(invalid_coord.shape[0]).cuda()
                    indices, distances = pointops.knn_query(1, valid_coord, valid_offset, invalid_coord, invalid_offset)
                    indices = indices[:, 0].cpu().numpy()
                    labels[invalid_label_mask] = labels[~invalid_label_mask][indices]

            
            # np.save(os.path.join(group_save_root, f"{str(scale)}.npy"), labels)
            save_path = os.path.join(save_root, f"{str(scale)}.ply")
            coord = input_dict["obj"]["coord"].cpu().numpy()
            random_color = []
            for i in range(max(labels) + 1):
                random_color.append(rgb_colors[i % len(rgb_colors)])
            random_color.append(np.array([0, 0, 0]))
            color = [random_color[i] for i in labels]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coord)
            pcd.colors = o3d.utility.Vector3dVector(color)
            o3d.io.write_point_cloud(save_path, pcd)

            labels = labels[pcd_inverse]

            # print(len(clusterer.labels_))
            self.logger.info(f"scale_{scale} has {max(labels)+1} groups")
            # print(min(clusterer.labels_))
            if self.mesh_voting:
                face_index = self.train_loader.face_index
                face_index = face_index[pcd_inverse]

                # Compute votes for each face using NumPy's bincount function
                # labels = clusterer.labels_
                num_faces = len(mesh.faces)
                num_labels = max(labels) + 1
                votes = np.zeros((num_faces, num_labels), dtype=np.int32)
                np.add.at(votes, (face_index, labels), 1)

                # Find the label with most votes for each face using NumPy's argmax function
                max_votes_labels = np.argmax(votes, axis=1)
                # Set the label to -1 for faces that have no corresponding points
                max_votes_labels[np.all(votes == 0, axis=1)] = -1

                valid_mask = max_votes_labels != -1
                face_centroids = mesh.triangles_center
                coord = torch.tensor(face_centroids).cuda().contiguous().float()
                valid_coord = coord[valid_mask]
                valid_offset = torch.tensor(valid_coord.shape[0]).cuda()
                invalid_coord = coord[~valid_mask]
                invalid_offset = torch.tensor(invalid_coord.shape[0]).cuda()
                indices, distances = pointops.knn_query(1, valid_coord, valid_offset, invalid_coord, invalid_offset)
                # # the first column is the point itself
                # indices = indices[:, 1].cpu().numpy()
                indices = indices[:, 0].cpu().numpy()
                mesh_group = max_votes_labels.copy()
                mesh_group[~valid_mask] = mesh_group[valid_mask][indices]

                np.save(os.path.join(group_save_root, f"mesh_{str(scale)}.npy"), mesh_group)

                # Assign color to each face based on the label with most votes
                for face, label in enumerate(mesh_group):
                    color = (random_color[label] * 255).astype(np.uint8)
                    color_with_alpha = np.append(color, 255)  # Add alpha value
                    mesh.visual.face_colors[face] = color_with_alpha

                # Save the new mesh
                mesh_save_path = os.path.join(save_root, f"mesh_{str(scale)}.ply")
                mesh.export(mesh_save_path)
                
    
    def _get_quantile_func(self, scales: torch.Tensor, distribution="normal"):
        """
        Use 3D scale statistics to normalize scales -- use quantile transformer.
        """
        scales = scales.flatten()
        max_grouping_scale = 2
        scales = scales[(scales > 0) & (scales < max_grouping_scale)]

        scales = scales.detach().cpu().numpy()

        # Calculate quantile transformer
        quantile_transformer = QuantileTransformer(output_distribution=distribution)
        quantile_transformer = quantile_transformer.fit(scales.reshape(-1, 1))

        def quantile_transformer_func(scales):
            # This function acts as a wrapper for QuantileTransformer.
            # QuantileTransformer expects a numpy array, while we have a torch tensor.
            return torch.Tensor(
                quantile_transformer.transform(scales.cpu().numpy())
            ).to(scales.device)

        return quantile_transformer_func

    def run_step(self):
        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        return writer

    def build_train_loader(self):
        self.cfg.data.train.oid = self.cfg.oid
        self.cfg.data.train.label = self.cfg.label
        train_data = build_dataset(self.cfg.data.train)
        return train_data

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        # self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        self.cfg.scheduler.total_steps = self.max_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        scaler = torch.cuda.amp.GradScaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader
