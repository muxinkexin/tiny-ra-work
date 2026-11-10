"""True-input-resolution Eigen evaluation; independent and base-anchored median scales."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import networks
from layers import disp_to_depth


class Images(Dataset):
    def __init__(self, root, lines, width, height, ext):
        self.root, self.lines, self.width, self.height, self.ext = Path(root), lines, width, height, ext
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, i):
        folder, frame, side = self.lines[i].split()
        side = {'l':2, 'r':3, '2':2, '3':3}[side]
        path = self.root/folder/f'image_0{side}'/'data'/f'{int(frame):010d}{self.ext}'
        with Image.open(path) as im:
            im = im.convert('RGB').resize((self.width, self.height), Image.Resampling.LANCZOS)
            arr = np.array(im, dtype=np.float32)/255.
        return torch.from_numpy(arr).permute(2,0,1)


def eigen_mask(gt):
    h,w = gt.shape
    mask = np.isfinite(gt) & (gt>1e-3) & (gt<80)
    y0,y1,x0,x1 = np.array([.40810811*h,.99189189*h,.03594771*w,.96405229*w]).astype(np.int32)
    crop = np.zeros_like(mask)
    crop[y0:y1,x0:x1] = True
    return mask & crop


def errors(gt, pred):
    r = np.maximum(gt/pred, pred/gt)
    return [np.mean(np.abs(gt-pred)/gt), np.mean((gt-pred)**2/gt),
            np.sqrt(np.mean((gt-pred)**2)), np.sqrt(np.mean((np.log(gt)-np.log(pred))**2)),
            np.mean(r<1.25), np.mean(r<1.25**2), np.mean(r<1.25**3)]


def load_model(folder, device):
    enc = torch.load(Path(folder)/'encoder.pth', map_location='cpu', weights_only=False)
    height, width = int(enc['height']), int(enc['width'])
    model = networks.LiteMono(model='lite-mono-tiny', height=height, width=width)
    dec = networks.TinyDepthDecoder(model.num_ch_enc, scales=[0])
    for module, raw in [(model,enc),(dec,torch.load(Path(folder)/'depth.pth', map_location='cpu', weights_only=False))]:
        expected = module.state_dict()
        chosen = {k:v for k,v in raw.items() if k in expected}
        module.load_state_dict(chosen, strict=True)
        module.to(device).eval()
    return model, dec, width, height


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default='/home/zy/KITTI')
    p.add_argument('--weights', required=True)
    p.add_argument('--test_files', default='splits/eigen/test_files.txt')
    p.add_argument('--gt', default='splits/eigen/gt_depths_tiny_ra.npz')
    p.add_argument('--sizes', nargs='+', default=['640x192','416x128','512x160','832x256','1024x320'])
    p.add_argument('--out', required=True)
    p.add_argument('--ext', choices=['.png','.jpg'], default='.png')
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--no_cuda', action='store_true')
    p.add_argument('--min_depth', type=float, default=.1)
    p.add_argument('--max_depth', type=float, default=100.)
    a = p.parse_args()
    cv2.setNumThreads(0)
    device = torch.device('cpu' if a.no_cuda else 'cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    lines = Path(a.test_files).read_text().splitlines()
    gtpack = np.load(a.gt, allow_pickle=True)
    gt = gtpack['data']
    if len(gt) != len(lines):
        raise ValueError('GT count differs from split count')
    if 'filenames' in gtpack and gtpack['filenames'].tolist() != lines:
        raise ValueError('GT filename order differs from test split')
    if 'filenames' not in gtpack:
        raise ValueError('Use tiny_ra_check.py --export_gt to create order-verified GT')
    encoder, decoder, basew, baseh = load_model(a.weights, device)
    sizes = [tuple(map(int, s.lower().split('x'))) for s in a.sizes]
    if len(sizes) != len(set(sizes)):
        raise ValueError('Duplicate requested sizes')
    sizes = [(basew,baseh)] + [s for s in sizes if s != (basew,baseh)]
    if any(w%16 or h%16 or min(w,h)<32 for w,h in sizes):
        raise ValueError('Evaluation sizes must be >=32 and divisible by 16; no implicit resize-to-training-size is allowed')
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new, empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    anchor = np.zeros(len(lines), dtype=np.float64)
    # A small common grid bounds RAM and defines the diagnostic bandwidth explicitly.
    base_grid = np.zeros((len(lines),64,192), dtype=np.float32)
    names = ['abs_rel','sq_rel','rmse','rmse_log','a1','a2','a3']
    summaries, individual = [], []
    for width,height in sizes:
        ds = Images(a.data_path, lines, width, height, a.ext)
        loader = DataLoader(ds, batch_size=a.batch_size, num_workers=a.workers, shuffle=False)
        dummy = torch.zeros(a.batch_size,3,height,width, device=device)
        for _ in range(2):
            decoder(encoder(dummy))
        sync(device)
        idx, model_time = 0, 0.
        independent, anchored, drift, shape, ratios = [], [], [], [], []
        for batch in loader:
            batch = batch.to(device)
            assert batch.shape[-2:] == (height,width)
            sync(device)
            start = time.perf_counter()
            pred = decoder(encoder(batch))[('disp',0)]
            sync(device)
            model_time += time.perf_counter()-start
            if pred.shape[-2:] != (height,width):
                raise RuntimeError('Unexpected output size')
            inv, _ = disp_to_depth(pred, a.min_depth, a.max_depth)
            common = (1/F.interpolate(inv, (64,192), mode='bilinear', align_corners=False)).cpu().numpy()[:,0]
            inv = inv.cpu().numpy()[:,0]
            for j in range(len(batch)):
                g = np.asarray(gt[idx], dtype=np.float64)
                mask = eigen_mask(g)
                if not mask.any():
                    raise ValueError(f'No valid GT pixels at index {idx}: {lines[idx]}')
                depth = 1/cv2.resize(inv[j], (g.shape[1],g.shape[0]), interpolation=cv2.INTER_LINEAR)
                g, depth = g[mask], depth[mask].astype(np.float64)
                ratio = np.median(g)/np.median(depth)
                if (width,height) == (basew,baseh):
                    anchor[idx] = ratio
                    base_grid[idx] = common[j]
                ei = errors(g, np.clip(depth*ratio,1e-3,80))
                ea = errors(g, np.clip(depth*anchor[idx],1e-3,80))
                logdiff = np.log(common[j])-np.log(base_grid[idx])
                d = float(abs(np.median(logdiff)))
                sh = float(np.mean(np.abs(logdiff-np.mean(logdiff))))
                independent.append(ei); anchored.append(ea); drift.append(d); shape.append(sh); ratios.append(ratio)
                for protocol, values in [('independent_median',ei),('base_anchored',ea)]:
                    individual.append(dict(index=idx,filename=lines[idx],width=width,height=height,protocol=protocol,
                                           **dict(zip(names,values)),scale_ratio=ratio,anchor_ratio=anchor[idx],
                                           log_scale_drift=d,log_shape_difference=sh))
                idx += 1
        if idx != len(lines):
            raise RuntimeError('Evaluation count mismatch')
        for protocol, values in [('independent_median',independent),('base_anchored',anchored)]:
            row = dict(width=width,height=height,protocol=protocol,n=idx,**dict(zip(names,np.mean(values,axis=0))),
                       log_scale_drift=float(np.mean(drift)),log_shape_difference=float(np.mean(shape)),
                       scale_ratio_median=float(np.median(ratios)),model_ms_per_image=1000*model_time/idx)
            summaries.append(row)
            print(json.dumps(row), flush=True)
    for filename, rows in [('summary.csv',summaries),('per_image.csv',individual)]:
        with (out/filename).open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metadata = dict(options=vars(a),actual_sizes=sizes,architecture_size=[basew,baseh],
                    parameter_count=sum(p.numel() for m in (encoder,decoder) for p in m.parameters()),
                    split_sha256=hashlib.sha256(Path(a.test_files).read_bytes()).hexdigest(),
                    encoder_sha256=hashlib.sha256((Path(a.weights)/'encoder.pth').read_bytes()).hexdigest(),
                    depth_sha256=hashlib.sha256((Path(a.weights)/'depth.pth').read_bytes()).hexdigest(),
                    common_grid_wh=[192,64],timing='two warm-up forwards per size; model-only wall time with device synchronization; no IO or resize',
                    torch=torch.__version__,cuda=torch.version.cuda,device=str(device),flip_postprocess=False)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))


if __name__ == '__main__':
    main()
