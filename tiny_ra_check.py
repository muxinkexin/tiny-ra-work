"""KITTI layout/neighbor audit, augmentation preview, or order-verified GT export."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw
import torch


def image_path(root, line, delta, ext):
    folder, frame, side = line.split()
    camera = {'l':2,'r':3,'2':2,'3':3}[side]
    return root/folder/f'image_0{camera}'/'data'/f'{int(frame)+delta:010d}{ext}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default='/home/zy/KITTI')
    p.add_argument('--ext', choices=['.png','.jpg'], default='.png')
    p.add_argument('--train_split', default='eigen_zhou')
    p.add_argument('--test_files', default='splits/eigen/test_files.txt')
    p.add_argument('--export_gt', action='store_true')
    p.add_argument('--import_gt', help='An existing GT archive that was generated from --import_test_files')
    p.add_argument('--import_test_files', help='The exact source split used to generate --import_gt')
    p.add_argument('--gt_out', default='splits/eigen/gt_depths_tiny_ra.npz')
    p.add_argument('--preview', default='augmentation_preview.png')
    a = p.parse_args()
    root = Path(a.data_path)
    if not root.is_dir():
        raise SystemExit(f'Cannot see the specified KITTI root: {root}')
    if a.import_gt:
        if not a.import_test_files:
            raise ValueError('--import_gt requires the exact --import_test_files used by that GT')
        lines = Path(a.test_files).read_text().splitlines()
        if Path(a.import_test_files).read_text().splitlines() != lines:
            raise ValueError('Source GT split differs in content/order from the destination split')
        pack = np.load(a.import_gt,allow_pickle=True)
        old = pack['data']
        if len(old) != len(lines):
            raise ValueError('Source GT count differs from split')
        if 'filenames' in pack and pack['filenames'].tolist() != lines:
            raise ValueError('Source GT embedded filename order differs from its declared split')
        if Path(a.gt_out).exists():
            raise FileExistsError('Choose a new --gt_out')
        Path(a.gt_out).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(a.gt_out,data=old,filenames=np.array(lines),
                            split_sha256=hashlib.sha256(Path(a.test_files).read_bytes()).hexdigest())
        print('Imported trusted GT; archive contents are assumed to follow the declared source split:',a.gt_out)
        return
    if a.export_gt:
        from kitti_utils import generate_depth_map
        lines = Path(a.test_files).read_text().splitlines()
        data = np.empty(len(lines), dtype=object)
        for i,line in enumerate(lines):
            folder, frame, side = line.split()
            camera = {'l':2,'r':3,'2':2,'3':3}[side]
            calib = root/folder.split('/')[0]
            velo = root/folder/'velodyne_points/data'/f'{int(frame):010d}.bin'
            for path in (calib/'calib_cam_to_cam.txt',calib/'calib_velo_to_cam.txt',velo):
                if not path.is_file():
                    raise FileNotFoundError(path)
            data[i] = generate_depth_map(str(calib),str(velo),camera,vel_depth=True).astype(np.float32)
            if (i+1)%100 == 0:
                print('Exported',i+1,'/',len(lines),flush=True)
        path = Path(a.gt_out)
        if path.exists():
            raise FileExistsError('GT output already exists; choose a new --gt_out')
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,data=data,filenames=np.array(lines),
                            split_sha256=hashlib.sha256(Path(a.test_files).read_bytes()).hexdigest())
        print('Saved order-verified GT:',path)
        return
    reports = {}
    train_lines = None
    for name,filename,deltas in [('train',f'splits/{a.train_split}/train_files.txt',(-1,0,1)),
                                 ('val',f'splits/{a.train_split}/val_files.txt',(-1,0,1)),
                                 ('test',a.test_files,(0,))]:
        lines = Path(filename).read_text().splitlines()
        if not lines:
            raise ValueError('Empty split: '+filename)
        if name == 'train':
            train_lines = lines
        missing = []
        for line in lines:
            for delta in deltas:
                path = image_path(root,line,delta,a.ext)
                if not path.is_file():
                    missing.append(str(path))
        reports[name] = {'samples':len(lines),'missing_images':len(missing),'examples':missing[:10],
                         'split_sha256':hashlib.sha256(Path(filename).read_bytes()).hexdigest()}
        print(name, json.dumps(reports[name]),flush=True)
    Path('preflight.json').write_text(json.dumps(reports,indent=2))
    if any(r['missing_images'] for r in reports.values()):
        raise SystemExit('Missing files: check Raw layout, root path, image suffix, and neighbor frames. Do not train on a silently filtered split.')
    from datasets.kitti_dataset import KITTIRAWDataset
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    ds = KITTIRAWDataset(str(root),train_lines,192,640,[0,-1,1],1,is_train=True,img_ext=a.ext)
    item = ds[0]
    canvas = Image.new('RGB',(640*3,192+30),'#f5f1e8')
    draw = ImageDraw.Draw(canvas)
    for i,name in enumerate(('HiS','MiS','LoS')):
        tensor = item[('color_'+name,0,0)]
        if tensor.shape != (3,192,640) or not torch.isfinite(tensor).all():
            raise RuntimeError('Invalid augmented image')
        im = Image.fromarray((tensor.permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8))
        canvas.paste(im,(i*640,30));draw.text((i*640+10,8),{'HiS':'High scale: resized + cropped','MiS':'Middle scale: reference','LoS':'Low scale: resized + stitched'}[name],fill='#18383b')
    canvas.save(a.preview)
    print('High resize and crop:',item['resize_HiS'].tolist(),item['dxy_HiS'].tolist())
    print('Low resize:',item['resize_LoS'].tolist())
    print('Augmentation preview saved:',a.preview)


if __name__ == '__main__':
    main()
