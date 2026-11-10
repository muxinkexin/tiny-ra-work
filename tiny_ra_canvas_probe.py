"""Frozen Lite-Mono tiny input-canvas diagnostic. No GT, training, or optimizer."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


PAIRS = [('B_vs_A','B','A'), ('C_vs_A','C','A'), ('D_vs_A','D','A'),
         ('E_vs_A','E','A'), ('C_vs_B','C','B'), ('C_vs_D','C','D'),
         ('C_vs_E','C','E')]


def size(value):
    try:
        w,h = map(int,value.lower().split('x'))
        if min(w,h)<=0:
            raise ValueError()
        return w,h
    except ValueError:
        raise argparse.ArgumentTypeError('Use positive WIDTHxHEIGHT, e.g. 480x144')


def canvases(low, height, width, fill=.45):
    """Exact RA tail-stitch, neutral constant, and edge replication."""
    _,h,w = low.shape
    if not (h<=height<=2*h and w<=width<=2*w):
        raise ValueError('RA stitching requires each low dimension between half and full native size')
    c = low.new_empty((3,height,width))
    py,px = 2*h-height,2*w-width
    c[:,:h,:w] = low
    c[:,h:,:w] = low[:,py:h,:w]
    c[:,:h,w:] = low[:,:h,px:w]
    c[:,h:,w:] = low[:,py:h,px:w]
    d = low.new_full((3,height,width),fill)
    d[:,:h,:w] = low
    e = F.pad(low.unsqueeze(0),(0,width-w,0,height-h),mode='replicate')[0]
    for image in [c,d,e]:
        if not torch.equal(image[:,:h,:w],low):
            raise AssertionError('Canvas valid pixels differ from real low-resolution input')
    return {'B':low,'C':c,'D':d,'E':e}


def pair_metrics(test, reference):
    x,y = np.asarray(test,dtype=np.float64),np.asarray(reference,dtype=np.float64)
    if x.shape!=y.shape or x.size==0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Aligned maps must have the same nonempty finite shape')
    if min(x.min(),y.min())<=0:
        raise ValueError('Depth must be positive')
    residual = np.log(x)-np.log(y)
    offset = float(np.median(residual))
    ratio = float(np.median(x)/np.median(y))
    return {'signed_log_scale':offset,'median_depth_ratio':ratio,
            'median_depth_change_pct':100*(ratio-1),
            'shape_log_mae':float(np.mean(np.abs(residual-offset))),
            'total_log_mae':float(np.mean(np.abs(residual)))}


def state_hash(*models):
    digest = hashlib.sha256()
    for i,model in enumerate(models):
        for key,value in model.state_dict().items():
            digest.update(f'{i}/{key}'.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def load_strict(model, checkpoint):
    expected = model.state_dict()
    for key,value in expected.items():
        if key not in checkpoint or not torch.is_tensor(checkpoint[key]) or checkpoint[key].shape!=value.shape:
            raise ValueError(f'Checkpoint mismatch: {key}')
    model.load_state_dict({k:checkpoint[k] for k in expected},strict=True)
    return sorted(set(checkpoint)-set(expected))


def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows to write')
    with open(path,'w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, bootstrap, seed):
    groups = {}
    for row in rows:
        groups.setdefault((row['model'],row['pair'],row['region']),[]).append(row)
    result = []
    for (model,pair,region),items in groups.items():
        logs = np.array([r['signed_log_scale'] for r in items])
        ratios = np.array([r['median_depth_ratio'] for r in items])
        drives = sorted({r['drive'] for r in items})
        lo,hi = None,None
        if bootstrap and len(drives)>1:
            ids = {d:i for i,d in enumerate(drives)}
            codes = np.array([ids[r['drive']] for r in items])
            counts = np.bincount(codes,minlength=len(drives))
            sums = np.bincount(codes,weights=logs,minlength=len(drives))
            draws = np.random.default_rng(seed).integers(0,len(drives),size=(bootstrap,len(drives)))
            samples = sums[draws].sum(1)/counts[draws].sum(1)
            lo,hi = map(float,np.quantile(samples,[.025,.975]))
        result.append({'model':model,'pair':pair,'region':region,
            'n_images':len(items),'n_drives':len(drives),
            'mean_signed_log_scale':float(logs.mean()),
            'median_signed_log_scale':float(np.median(logs)),
            'mean_abs_signed_log_scale':float(np.mean(np.abs(logs))),
            'median_depth_ratio':float(np.median(ratios)),
            'median_depth_change_pct':float(np.median(100*(ratios-1))),
            'positive_depth_ratio_fraction':float(np.mean(ratios>1)),
            'mean_shape_log_mae':float(np.mean([r['shape_log_mae'] for r in items])),
            'mean_total_log_mae':float(np.mean([r['total_log_mae'] for r in items])),
            'ci95_mean_log_scale_low':lo,'ci95_mean_log_scale_high':hi})
    return result


def self_test():
    for h,w,H,W in [(3,6,4,8),(2,4,4,8),(4,8,4,8)]:
        low=torch.arange(3*h*w,dtype=torch.float32).reshape(3,h,w)/(3*h*w)
        images=canvases(low,H,W)
        assert all(torch.equal(v[:,:h,:w],low) for v in images.values())
        py,px=2*h-H,2*w-W
        assert torch.equal(images['C'][:,h:,w:],low[:,py:h,px:w])
        if H==h and W==w:
            assert all(torch.equal(v,low) for v in images.values())
    x=np.arange(1,25,dtype=np.float64).reshape(4,6)
    r=pair_metrics(1.2*x,x)
    assert np.isclose(r['median_depth_ratio'],1.2)
    assert np.isclose(r['signed_log_scale'],np.log(1.2))
    assert r['shape_log_mae']<1e-14
    assert pair_metrics(x,x)['total_log_mae']==0
    changed=x.copy();changed[:2]*=1.5
    assert pair_metrics(changed,x)['shape_log_mae']>0
    for a,b in [(x[:0],x[:0]),(-x,x)]:
        try:pair_metrics(a,b)
        except ValueError:pass
        else:raise AssertionError('Invalid maps accepted')
    try:canvases(torch.zeros(3,1,2),4,8)
    except ValueError:pass
    else:raise AssertionError('Invalid RA size accepted')
    items=[]
    for i in range(4):
        items.append({'model':'test','pair':'X_vs_Y','region':'full','drive':str(i//2),
                      'signed_log_scale':.1,'median_depth_ratio':float(np.exp(.1)),
                      'shape_log_mae':0.,'total_log_mae':.1})
    summary=summarize(items,1000,42)[0]
    assert np.isclose(summary['ci95_mean_log_scale_low'],.1)
    assert np.isclose(summary['ci95_mean_log_scale_high'],.1)
    print('PASS: exact shared pixels, exact RA tail stitching, half/full-size limits, scale/shape metrics, invalid input guards')


@torch.inference_mode()
def run(args, parser):
    W,H=args.native;w,h=args.low
    if min(W,H,w,h)<32 or any(v%16 for v in [W,H,w,h]) or w*H!=h*W:
        parser.error('Native/low sizes must share aspect ratio and be divisible by 16')
    if not (h<=H<=2*h and w<=W<=2*w) or 2*args.margin>=min(w,h):
        parser.error('Invalid low/native size or interior margin')
    if args.max_images<0 or args.margin<0 or args.bootstrap<0:
        parser.error('max_images, margin, bootstrap must be nonnegative')
    if not 0<args.min_depth<args.max_depth or not 0<=args.drop_path_rate<1:
        parser.error('Invalid depth range/drop_path_rate')
    out=Path(args.out).resolve();runs=Path(args.runs).resolve()
    if out.exists():parser.error('Output exists; use a new output directory')
    lines=[s.strip() for s in Path(args.split_file).read_text().splitlines() if s.strip()]
    if args.max_images:lines=lines[:args.max_images]
    if not lines:parser.error('Empty split')
    if len(set(lines))!=len(lines):parser.error('Duplicate split entries')
    paths=[]
    for line in lines:
        folder,frame,side=line.split()
        camera={'l':'02','r':'03','2':'02','3':'03'}[side]
        path=Path(args.data_path)/folder/f'image_{camera}'/'data'/f'{int(frame):010d}{args.ext}'
        if not path.is_file():raise FileNotFoundError(path)
        paths.append(path)
    weight_dirs={m:runs/f'tiny_{m}_s{args.seed}'/'models'/f'weights_{args.epoch}' for m in args.models}
    for folder in weight_dirs.values():
        if out==folder or folder in out.parents or out in folder.parents:
            parser.error('Output must be separate from checkpoint directories')
        for name in ['encoder.pth','depth.pth']:
            if not (folder/name).is_file():raise FileNotFoundError(folder/name)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(Path.cwd()))
    import networks
    decoder_cls=getattr(networks,'TinyDepthDecoder',None)
    if decoder_cls is None:decoder_cls=getattr(networks,'DepthDecoder',None)
    if decoder_cls is None:raise AttributeError('networks needs TinyDepthDecoder or DepthDecoder')
    out.mkdir(parents=True);(out/'preview_inputs').mkdir()
    metadata={'status':'running','arguments':vars(args),'torch_version':torch.__version__,
        'n_images':len(lines),'split_sha256':hashlib.sha256('\n'.join(lines).encode()).hexdigest(),
        'pair_definition':'X_vs_Y: test X minus reference Y; all outputs compared on the same low-resolution grid',
        'scale_definition':'median_depth_ratio=median(Ztest)/median(Zref); signed_log_scale=median(log(Ztest)-log(Zref))',
        'shape_definition':'mean(abs(log(Ztest)-log(Zref)-signed_log_scale))',
        'limitations':'No GT. Dense full/interior ROI differs from the sparse GT evaluation mask. CI is conditional on one checkpoint.',
        'models':{}}
    meta_path=out/'metadata.json'
    meta_path.write_text(json.dumps(metadata,indent=2)+'\n')
    rows=[]
    for model,folder in weight_dirs.items():
        enc=torch.load(folder/'encoder.pth',map_location='cpu',weights_only=False)
        if (int(enc.get('width',W)),int(enc.get('height',H)))!=(W,H):
            raise ValueError('Native reference size must match checkpoint metadata')
        encoder=networks.LiteMono(model='lite-mono-tiny',height=H,width=W,drop_path_rate=args.drop_path_rate)
        decoder=decoder_cls(encoder.num_ch_enc,scales=[0])
        ignored_e=load_strict(encoder,enc)
        ignored_d=load_strict(decoder,torch.load(folder/'depth.pth',map_location='cpu',weights_only=False))
        for module in [encoder,decoder]:module.to(args.device).eval().requires_grad_(False)
        assert all(not m.training for module in [encoder,decoder] for m in module.modules())
        before=state_hash(encoder,decoder)
        repeat_difference=None;repeat_metrics=None
        def inverse_depth(image):
            q=decoder(encoder(image.unsqueeze(0).to(args.device)))[('disp',0)]
            if q.shape[-2:]!=image.shape[-2:] or not torch.isfinite(q).all():
                raise ValueError('Invalid decoder output')
            return (1/args.max_depth+(1/args.min_depth-1/args.max_depth)*q).cpu()
        for index,(line,path) in enumerate(zip(lines,paths)):
            with Image.open(path) as raw:
                raw=raw.convert('RGB')
                def resize(width,height):
                    return torch.from_numpy(np.asarray(raw.resize((width,height),Image.Resampling.LANCZOS),dtype=np.float32).copy()).permute(2,0,1)/255.
                a,low=resize(W,H),resize(w,h)
            images={'A':a,**canvases(low,H,W)}
            if model==args.models[0] and index==0:
                for name,image in images.items():
                    array=(image.permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)
                    Image.fromarray(array).save(out/'preview_inputs'/f'{name}.png')
            inverse={name:inverse_depth(image) for name,image in images.items()}
            if index==0:
                repeated=inverse_depth(a)
                repeat_difference=float((repeated-inverse['A']).abs().max())
            maps={}
            for name,value in inverse.items():
                if name=='A':value=F.interpolate(value,size=(h,w),mode='bilinear',align_corners=False)
                else:value=value[:,:,:h,:w]
                maps[name]=(1/value[0,0]).numpy()
            if index==0:
                repeated=1/F.interpolate(repeated,size=(h,w),mode='bilinear',align_corners=False)[0,0]
                repeat_metrics=pair_metrics(repeated.numpy(),maps['A'])
            for pair,test,reference in PAIRS:
                for region,margin in [('full',0),('interior',args.margin)]:
                    roi=(slice(margin,h-margin),slice(margin,w-margin))
                    row={'model':model,'index':index,'filename':line,'drive':line.split()[0],
                         'pair':pair,'region':region,'n_pixels':(h-2*margin)*(w-2*margin)}
                    row.update(pair_metrics(maps[test][roi],maps[reference][roi]))
                    rows.append(row)
            if (index+1)%100==0 or index+1==len(lines):
                print(f'{model}: {index+1}/{len(lines)}',flush=True)
        after=state_hash(encoder,decoder)
        if before!=after:raise AssertionError('Model state changed during frozen inference')
        metadata['models'][model]={'weights':str(folder),'encoder_sha256':file_hash(folder/'encoder.pth'),
            'depth_sha256':file_hash(folder/'depth.pth'),'ignored_encoder_entries':ignored_e,
            'ignored_depth_entries':ignored_d,'state_hash_before':before,'state_hash_after':after,
            'reference_repeat_max_inverse_depth_diff':repeat_difference,
            'reference_repeat_log_metrics':repeat_metrics}
        del encoder,decoder
    write_csv(out/'per_image.csv',rows)
    write_csv(out/'summary.csv',summarize(rows,args.bootstrap,args.seed))
    metadata['status']='complete';metadata['n_per_image_rows']=len(rows)
    meta_path.write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Complete: {out}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self_test',action='store_true')
    parser.add_argument('--data_path');parser.add_argument('--runs');parser.add_argument('--out')
    parser.add_argument('--split_file',default='splits/eigen/test_files.txt')
    parser.add_argument('--models',nargs='+',choices=['base','repeat','aug','ra'],default=['base','repeat','aug','ra'])
    parser.add_argument('--native',type=size,default=(640,192));parser.add_argument('--low',type=size,default=(480,144))
    parser.add_argument('--ext',default='.jpg');parser.add_argument('--epoch',type=int,default=19)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--device',default='cuda')
    parser.add_argument('--max_images',type=int,default=0);parser.add_argument('--margin',type=int,default=16)
    parser.add_argument('--bootstrap',type=int,default=5000)
    parser.add_argument('--min_depth',type=float,default=.1);parser.add_argument('--max_depth',type=float,default=100.)
    parser.add_argument('--drop_path_rate',type=float,default=.2)
    args=parser.parse_args()
    if args.self_test:self_test();return
    if not all([args.data_path,args.runs,args.out]):parser.error('Require --data_path --runs --out')
    if len(set(args.models))!=len(args.models):parser.error('Duplicate model names')
    run(args,parser)


if __name__=='__main__':
    main()
