"""Run from the patched RA-Depth root. Only monocular, separate-ResNet pose is supported."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from options import MonodepthOptions
from trainer import Trainer
from layers import transformation_from_parameters
from tiny_ra_core import objective


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def load_component(model, filename, encoder=False):
    raw = torch.load(filename, map_location='cpu', weights_only=False)
    if encoder and ('height' in raw or 'width' in raw):
        if (int(raw.get('width',640)),int(raw.get('height',192))) != (640,192):
            raise ValueError('This controlled transfer requires a 640x192 baseline checkpoint')
    if 'model' in raw and isinstance(raw['model'], dict):
        raw = raw['model']
    expected = model.state_dict()
    selected = {k: v for k, v in raw.items() if k in expected and torch.is_tensor(v) and v.shape == expected[k].shape}
    missing = set(expected) - set(selected)
    # ImageNet encoder has no depth-task-only position projection if keys differ;
    # only explicitly requested ImageNet transfer may allow missing keys below.
    if missing:
        raise RuntimeError(f'{filename}: missing/incompatible keys: {sorted(missing)[:12]}')
    model.load_state_dict(selected, strict=True)
    print(f'Loaded {filename}: {len(selected)} entries; ignored {len(raw)-len(selected)} metadata/extra entries')


class TinyRATrainer(Trainer):
    def __init__(self, opt):
        if opt.scales != [0] or opt.frame_ids != [0, -1, 1]:
            raise ValueError('This recipe requires --scales 0 and --frame_ids 0 -1 1')
        if opt.height != 192 or opt.width != 640:
            raise ValueError('Training is fixed at width=640, height=192')
        if opt.pose_model_type != 'separate_resnet' or opt.pose_model_input != 'pairs':
            raise ValueError('Only separate_resnet + pairs is supported')
        if any([opt.use_stereo, opt.predictive_mask, opt.v1_multiscale, opt.avg_reprojection,
                opt.disable_automasking, opt.no_ssim, opt.load_weights_folder]):
            raise ValueError('Unsupported upstream ablation/loading flag; use --init_folder or --resume')
        self.directory = Path(opt.log_dir) / opt.model_name
        if any((self.directory/name).exists() for name in ('last.pt','train.csv')) and not opt.resume:
            raise FileExistsError('Existing experiment: choose another --model_name or use --resume')
        if opt.init_folder and opt.encoder_pretrain:
            raise ValueError('--init_folder and --encoder_pretrain are mutually exclusive')
        if not any([opt.init_folder, opt.encoder_pretrain, opt.resume, opt.allow_random_init]):
            raise ValueError('Supply --init_folder or --encoder_pretrain; random tiny initialization requires --allow_random_init')
        super().__init__(opt)
        if opt.init_folder:
            for name, model in self.models.items():
                load_component(model, Path(opt.init_folder)/(name+'.pth'), encoder=(name=='encoder'))
        elif opt.encoder_pretrain:
            raw = torch.load(opt.encoder_pretrain, map_location='cpu', weights_only=False)
            raw = raw.get('model', raw)
            expected = self.models['encoder'].state_dict()
            selected = {k:v for k,v in raw.items() if k in expected and torch.is_tensor(v) and v.shape == expected[k].shape}
            if not selected:
                raise ValueError('No ImageNet encoder weights matched')
            coverage = sum(v.numel() for v in selected.values()) / sum(v.numel() for v in expected.values())
            if coverage < .95:
                raise ValueError(f'Only {coverage:.1%} of the encoder state matched; verify tiny variant and checkpoint format')
            missing, extra = self.models['encoder'].load_state_dict(selected, strict=False)
            print('ImageNet transfer matched', len(selected), 'missing', list(missing), 'unexpected', list(extra))
            (self.directory/'imagenet_transfer.json').write_text(json.dumps({'matched':len(selected), 'missing':list(missing)}, indent=2))
        self.start_epoch, self.step = 0, 0
        if opt.resume:
            ckpt = torch.load(opt.resume, map_location='cpu', weights_only=False)
            for key in ('mode', 'seed', 'batch_size', 'num_epochs', 'learning_rate', 'beta', 'scheduler_step_size', 'weights_init', 'png', 'disparity_smoothness', 'min_depth', 'max_depth', 'max_batches', 'num_workers', 'data_path', 'split'):
                if ckpt['options'][key] != vars(opt)[key]:
                    raise ValueError(f'Resume mismatch for {key}')
            for name, model in self.models.items():
                model.load_state_dict(ckpt['models'][name], strict=True)
            self.model_optimizer.load_state_dict(ckpt['optimizer'])
            self.model_lr_scheduler.load_state_dict(ckpt['scheduler'])
            self.start_epoch, self.step = ckpt['next_epoch'], ckpt['step']
            random.setstate(ckpt['python_rng'])
            np.random.set_state(ckpt['numpy_rng'])
            torch.set_rng_state(ckpt['torch_rng'])
            if torch.cuda.is_available() and ckpt['cuda_rng'] is not None:
                torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        self.directory.mkdir(parents=True, exist_ok=True)
        run = {'options': vars(opt), 'torch':torch.__version__, 'cuda':torch.version.cuda,
               'gpu':torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU',
               'precision':'FP32; TF32 disabled', 'patch':json.loads(Path('tiny_ra_manifest.json').read_text())}
        for split in ('train', 'val'):
            p = Path('splits')/opt.split/(split+'_files.txt')
            run[split+'_split_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
        run['initialization_sha256'] = {}
        if opt.init_folder:
            for name in self.models:
                p = Path(opt.init_folder)/(name+'.pth')
                run['initialization_sha256'][name] = hashlib.sha256(p.read_bytes()).hexdigest()
        elif opt.encoder_pretrain:
            run['initialization_sha256']['encoder_pretrain'] = hashlib.sha256(Path(opt.encoder_pretrain).read_bytes()).hexdigest()
        elif opt.resume:
            run['initialization_sha256']['resume'] = hashlib.sha256(Path(opt.resume).read_bytes()).hexdigest()
        (self.directory/'run.json').write_text(json.dumps(run, indent=2))

    def process_batch(self, inputs):
        inputs = {k:v.to(self.device, non_blocking=True) for k,v in inputs.items()}
        names = ['MiS'] if self.opt.mode == 'base' else ['HiS', 'MiS', 'LoS']
        disparities = {}
        for name in names:
            source_name = 'MiS' if self.opt.mode == 'repeat' else name
            # The upstream recipe jitters only the medium depth input.
            key = 'color_MiS_aug' if source_name == 'MiS' else 'color_'+source_name
            out = self.models['depth'](self.models['encoder'](inputs[(key, 0, 0)]))
            disparities[name] = out[('disp', 0)]
        poses = {}
        for f in (-1, 1):
            pair = [inputs[('color_MiS', j, 0)] for j in ((f,0) if f < 0 else (0,f))]
            feats = self.models['pose_encoder'](torch.cat(pair, 1))
            aa, tr = self.models['pose']([feats])
            poses[f] = transformation_from_parameters(aa[:,0], tr[:,0], invert=(f < 0))
        losses = objective(inputs, disparities, poses, self.ssim, self.opt.mode,
                           beta=self.opt.beta, min_depth=self.opt.min_depth,
                           max_depth=self.opt.max_depth, smooth_weight=self.opt.disparity_smoothness)
        return disparities, losses

    def save_checkpoint(self, next_epoch):
        self.epoch = next_epoch-1
        self.save_model()  # standard encoder/depth/pose files for evaluation
        state = {'models':{k:m.state_dict() for k,m in self.models.items()},
                 'optimizer':self.model_optimizer.state_dict(), 'scheduler':self.model_lr_scheduler.state_dict(),
                 'next_epoch':next_epoch, 'step':self.step, 'options':vars(self.opt),
                 'python_rng':random.getstate(), 'numpy_rng':np.random.get_state(),
                 'torch_rng':torch.get_rng_state(),
                 'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        temp = self.directory/'last.tmp.pt'
        torch.save(state, temp)
        os.replace(temp, self.directory/'last.pt')

    @torch.no_grad()
    def validation_proxy(self):
        self.set_eval()
        py, npstate, ts = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        seed_all(self.opt.seed+10000)
        loader = DataLoader(self.val_loader.dataset, batch_size=self.opt.batch_size,
                            shuffle=False, num_workers=0, drop_last=True)
        values = []
        for i, batch in enumerate(loader):
            if i >= self.opt.val_batches:
                break
            _, losses = self.process_batch(batch)
            values.append(float(losses['loss']))
        random.setstate(py)
        np.random.set_state(npstate)
        torch.set_rng_state(ts)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        self.set_train()
        if values:
            self.writers['val'].add_scalar('proxy_loss', float(np.mean(values)), self.step)
        return float(np.mean(values)) if values else None

    def train(self):
        start = time.time()
        for epoch in range(self.start_epoch, self.opt.num_epochs):
            self.epoch = epoch
            self.set_train()
            # Epoch-addressable ordering and worker RNG enable epoch-boundary resume.
            generator = torch.Generator().manual_seed(self.opt.seed+epoch)
            loader = DataLoader(self.train_loader.dataset, batch_size=self.opt.batch_size,
                                shuffle=True, num_workers=self.opt.num_workers,
                                pin_memory=self.device.type=='cuda', drop_last=True,
                                worker_init_fn=worker_seed, generator=generator)
            if not len(loader):
                raise ValueError('Training split has fewer samples than the batch size')
            for batch_index, inputs in enumerate(loader):
                if self.opt.max_batches and batch_index >= self.opt.max_batches:
                    break
                self.model_optimizer.zero_grad(set_to_none=True)
                _, losses = self.process_batch(inputs)
                if not torch.isfinite(losses['loss']):
                    raise FloatingPointError('Nonfinite training loss')
                losses['loss'].backward()
                checks = [torch.isfinite(p.grad).all() for p in self.parameters_to_train if p.grad is not None]
                if checks and not torch.stack(checks).all():
                    raise FloatingPointError('Nonfinite gradients')
                self.model_optimizer.step()
                if self.step % self.opt.log_frequency == 0:
                    row = {k:float(v.detach()) for k,v in losses.items()}
                    row.update(epoch=epoch, step=self.step, batch=batch_index,
                               lr=self.model_optimizer.param_groups[0]['lr'], elapsed_s=time.time()-start)
                    path = self.directory/'train.csv'
                    exists = path.exists()
                    with path.open('a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=list(row))
                        if not exists:
                            writer.writeheader()
                        writer.writerow(row)
                    for k,v in row.items():
                        self.writers['train'].add_scalar(k, v, self.step)
                    print(json.dumps(row), flush=True)
                self.step += 1
            self.model_lr_scheduler.step()
            if self.opt.val_batches:
                print('validation proxy', self.validation_proxy(), flush=True)
            if (epoch+1) % self.opt.save_frequency == 0 or epoch+1 == self.opt.num_epochs:
                self.save_checkpoint(epoch+1)
        for writer in self.writers.values():
            writer.close()


def main():
    options = MonodepthOptions()
    p = options.parser
    p.add_argument('--mode', choices=['base','repeat','aug','ra'], default='ra')
    p.add_argument('--beta', type=float, default=1.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--encoder_pretrain')
    p.add_argument('--init_folder')
    p.add_argument('--resume')
    p.add_argument('--allow_random_init', action='store_true', help='Synthetic smoke tests only; not the intended reproduction initialization')
    p.add_argument('--max_batches', type=int, default=0, help='Smoke test only; 0 means full epoch')
    p.add_argument('--val_batches', type=int, default=2)
    opt = options.parse()
    seed_all(opt.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    TinyRATrainer(opt).train()


if __name__ == '__main__':
    main()
