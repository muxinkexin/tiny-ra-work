"""Lite-Mono tiny forward-randomness and encoder-BN diagnostics.

Run from the tiny-ra-work directory. Original checkpoints are never overwritten.
BN calibration is an approximate average of fixed-size training-batch statistics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler


def seed_diagnostic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class _IsolatedItems(Dataset):
    def __init__(self, source, seed):
        self.source,self.seed,self.round = source,seed,0

    def __len__(self):
        return len(self.source)

    def __getattr__(self, name):
        return getattr(self.source,name)

    def __getitem__(self, index):
        key = f'{self.seed}/{self.round}/{index}'.encode()
        local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8],'little') % (2**63-1)
        py_state,np_state,cpu_state = random.getstate(),np.random.get_state(),torch.get_rng_state()
        try:
            random.seed(local_seed)
            np.random.seed(local_seed % (2**32))
            torch.set_rng_state(torch.Generator().manual_seed(local_seed).get_state())
            return self.source[index]
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(cpu_state)


class _DiagnosticLoader(DataLoader):
    def __iter__(self):
        self.dataset.round = getattr(self,'_diag_round',0)
        self._diag_round = self.dataset.round+1
        return super().__iter__()


def diagnostic_loader(loader, seed):
    """Standard loaders, CPU transforms, no workers, isolated per-item data RNG.

    Fresh short runs only. Resuming additionally requires the independent
    generator state AND iterator counter saved/restored by the trainer.
    """
    if loader.batch_size is None or type(loader.sampler) not in (RandomSampler,SequentialSampler):
        raise ValueError('Diagnostic loader only supports standard batched random/sequential sampling')
    if isinstance(loader.sampler,RandomSampler) and loader.sampler.replacement:
        raise ValueError('Replacement sampling is not supported by this diagnostic wrapper')
    if isinstance(loader.dataset,_IsolatedItems):
        raise ValueError('Diagnostic loader is already installed')
    return _DiagnosticLoader(_IsolatedItems(loader.dataset,seed),batch_size=loader.batch_size,
                      shuffle=isinstance(loader.sampler,RandomSampler),
                      num_workers=0,drop_last=loader.drop_last,
                      collate_fn=loader.collate_fn,pin_memory=loader.pin_memory,
                      generator=torch.Generator().manual_seed(seed),
                      persistent_workers=False)


def automask_noise(reference, seed, step, branch):
    """Explicitly pass the same step/branch identifiers in every compared run.

    Use in place of randn()*1e-5 at the automask tie-breaking site only.
    The caller must provide its actual global update index.
    """
    key = f'{seed}/{step}/{branch}'.encode()
    local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8],'little') % (2**63-1)
    generator = torch.Generator(device=reference.device).manual_seed(local_seed)
    return torch.randn(reference.shape,device=reference.device,
                       dtype=reference.dtype,generator=generator)*1e-5


def drop_modules(model):
    return [m for m in model.modules() if m.__class__.__name__ == 'DropPath']


def bn_modules(model):
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def apply_train_controls(models):
    """Call immediately AFTER EVERY trainer set_train()/model.train().

    Controls depth-encoder BN and DropPath only; pose architecture is unchanged.
    BN affine weight/bias remain trainable. This function never calls eval()
    on the whole encoder and never disables autograd.
    """
    encoder = models['encoder']
    if os.environ.get('TINY_RA_DP0') == '1':
        for m in drop_modules(encoder):
            m.drop_prob = 0.0
    if os.environ.get('TINY_RA_BN_FROZEN') == '1':
        for m in bn_modules(encoder):
            if m.running_mean is None or m.running_var is None:
                raise ValueError('Frozen BN requires existing running statistics')
            m.eval()


def install_train_controls(models):
    """Install once after constructing the current eager/state_dict trainer.

    Reapply controls after encoder.train(True), including calls from set_train().
    Use a fresh trainer process for each experimental configuration.
    """
    encoder = models['encoder']
    if getattr(encoder, '_tiny_ra_diag_installed', False):
        raise ValueError('Diagnostic controls are already installed')
    original_train = encoder.train

    def controlled_train(mode=True):
        result = original_train(mode)
        if mode:
            apply_train_controls(models)
        return result

    encoder.train = controlled_train
    encoder._tiny_ra_diag_installed = True
    if encoder.training:
        apply_train_controls(models)


def load_state(model, checkpoint, label):
    expected = model.state_dict()
    missing = [k for k in expected if k not in checkpoint]
    bad = [k for k in expected if k in checkpoint and
           (not torch.is_tensor(checkpoint[k]) or checkpoint[k].shape != expected[k].shape)]
    if missing or bad:
        raise ValueError(f'{label}: missing={missing[:6]}, mismatched={bad[:6]}')
    model.load_state_dict({k: checkpoint[k] for k in expected}, strict=True)
    return sorted(set(checkpoint)-set(expected))


class RawTargets(Dataset):
    def __init__(self, root, split_file, ext, height, width, count, seed):
        lines = [s.strip() for s in Path(split_file).read_text().splitlines() if s.strip()]
        random.Random(seed).shuffle(lines)
        self.lines = lines[:count]
        self.root, self.ext = Path(root), ext
        self.height, self.width = height, width
        if len(self.lines) < count:
            raise ValueError(f'Need {count} split entries, found {len(self.lines)}')

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        folder, frame, side = self.lines[index].split()
        camera = {'l':'02', 'r':'03', '2':'02', '3':'03'}[side]
        path = self.root/folder/f'image_{camera}'/'data'/f'{int(frame):010d}{self.ext}'
        with Image.open(path) as im:
            im = im.convert('RGB').resize((self.width,self.height), Image.Resampling.LANCZOS)
            array = np.asarray(im, dtype=np.float32).copy()/255.0
        return torch.from_numpy(array).permute(2,0,1)


@torch.no_grad()
def forward_probe(encoder, decoder, image, repeats, seed):
    if repeats < 2:
        raise ValueError('repeats must be at least 2')
    report = {}
    for mode in ['eval', 'only_droppath_train', 'train_dp_off_bn_frozen']:
        encoder.eval()
        decoder.eval()
        if mode == 'only_droppath_train':
            for m in drop_modules(encoder):
                m.train()
        elif mode == 'train_dp_off_bn_frozen':
            encoder.train()
            for m in bn_modules(encoder):
                m.eval()
            for m in encoder.modules():
                if m.__class__.__name__ == 'DropPath' or isinstance(m, nn.Dropout):
                    m.eval()
        assert all(not m.training for m in bn_modules(encoder))
        before = {k:v.detach().clone() for k,v in encoder.state_dict().items()}
        seed_diagnostic(seed)  # Once per condition, never once per forward.
        reference = None
        differences = []
        for _ in range(repeats):
            prediction = decoder(encoder(image))[('disp',0)]
            if not torch.isfinite(prediction).all():
                raise ValueError('Non-finite prediction')
            if reference is None:
                reference = prediction.clone()
            differences.append((prediction-reference).abs())
        values = torch.stack(differences[1:])
        report[mode] = {'mean_abs_diff':values.mean().item(),
                        'max_abs_diff':values.max().item()}
        assert all(torch.equal(v,encoder.state_dict()[k]) for k,v in before.items())
    encoder.eval()
    return report


@torch.no_grad()
def calibrate_encoder(encoder, loader, device):
    modules = bn_modules(encoder)
    if not modules:
        raise ValueError('No encoder BatchNorm found')
    if any(not m.track_running_stats for m in modules):
        raise ValueError('Calibration requires track_running_stats=True')
    before = {k:v.detach().cpu().clone() for k,v in encoder.named_parameters()}
    old_momenta = [m.momentum for m in modules]
    encoder.eval()  # Keep DropPath/Dropout OFF.
    for m in modules:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    batches = 0
    try:
        for image in loader:
            encoder(image.to(device))
            batches += 1
    finally:
        for m,momentum in zip(modules,old_momenta):
            m.momentum = momentum
        encoder.eval()
    if batches == 0:
        raise ValueError('Calibration loader produced no batches')
    assert all(torch.equal(before[k],v.detach().cpu()) for k,v in encoder.named_parameters())
    assert all(int(m.num_batches_tracked)==batches for m in modules)
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['probe','calibrate'],required=True)
    parser.add_argument('--weights',required=True)
    parser.add_argument('--data_path',required=True)
    parser.add_argument('--split_file',default='splits/eigen_zhou/train_files.txt')
    parser.add_argument('--ext',default='.jpg')
    parser.add_argument('--out',required=True,help='New directory; must not exist')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch_size',type=int,default=12)
    parser.add_argument('--batches',type=int,default=200)
    parser.add_argument('--repeats',type=int,default=8)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--drop_path_rate',type=float,default=0.2,
                        help='Match the encoder constructor used during training')
    args = parser.parse_args()
    if args.batch_size < 1 or args.batches < 1 or args.repeats < 2:
        parser.error('Require batch_size>=1, batches>=1, repeats>=2')
    if not 0 <= args.drop_path_rate < 1:
        parser.error('Require 0 <= drop_path_rate < 1')
    source,out = Path(args.weights).resolve(),Path(args.out).resolve()
    if out.exists() or out == source or source in out.parents or out in source.parents:
        parser.error('out must be a new, non-nested directory separate from weights')
    seed_diagnostic(args.seed)
    sys.path.insert(0,str(Path.cwd()))
    import networks
    enc_ckpt = torch.load(source/'encoder.pth',map_location='cpu',weights_only=False)
    height,width = int(enc_ckpt.get('height',192)),int(enc_ckpt.get('width',640))
    encoder = networks.LiteMono(model='lite-mono-tiny',height=height,width=width,
                               drop_path_rate=args.drop_path_rate)
    ignored_encoder = load_state(encoder,enc_ckpt,'encoder')
    encoder.to(args.device)
    count = args.batch_size * (args.batches if args.action=='calibrate' else 1)
    dataset = RawTargets(args.data_path,args.split_file,args.ext,height,width,count,args.seed)
    loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=0,drop_last=True)
    report = dict(vars(args))
    report.update(height=height,width=width,torch_version=torch.__version__,
                  encoder_bn_count=len(bn_modules(encoder)),droppath_count=len(drop_modules(encoder)),
                  droppath_probabilities=[m.drop_prob for m in drop_modules(encoder)],
                  ignored_encoder_entries=ignored_encoder,
                  selected_split_sha256=hashlib.sha256('\n'.join(dataset.lines).encode()).hexdigest())
    if args.action=='probe':
        depth_ckpt = torch.load(source/'depth.pth',map_location='cpu',weights_only=False)
        decoder = networks.TinyDepthDecoder(encoder.num_ch_enc,scales=[0])
        report['ignored_depth_entries'] = load_state(decoder,depth_ckpt,'depth')
        decoder.to(args.device)
        report['forward_probe'] = forward_probe(encoder,decoder,next(iter(loader)).to(args.device),args.repeats,args.seed)
        out.mkdir(parents=True)
    else:
        report['calibration_batches'] = calibrate_encoder(encoder,loader,args.device)
        report['calibration_note'] = 'Equal-size batch-statistics average; not exact global population statistics.'
        original = enc_ckpt
        recalibrated = dict(original)
        recalibrated.update({k:v.detach().cpu() for k,v in encoder.state_dict().items()})
        allowed = {f'{name}.{suffix}' for name,m in encoder.named_modules()
                   if isinstance(m,nn.modules.batchnorm._BatchNorm)
                   for suffix in ['running_mean','running_var','num_batches_tracked']}
        changed = [k for k in encoder.state_dict() if not torch.equal(original[k],recalibrated[k])]
        if not set(changed).issubset(allowed):
            raise AssertionError('Unexpected change outside BN running buffers')
        report['changed_encoder_entries'] = changed
        shutil.copytree(source,out)
        torch.save(recalibrated,out/'encoder.pth')
    (out/'bn_dp_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
