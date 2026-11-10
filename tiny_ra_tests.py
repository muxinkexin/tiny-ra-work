"""Meaningful CPU/GPU checks for alignment, masks, gradients, and true input sizes."""
import argparse
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from layers import SSIM
import networks
from tiny_ra_core import shared_jitter, warp, consistency, objective


def metadata(b,h=192,w=640):
    K = torch.eye(4).repeat(b,1,1)
    K[:,0,0],K[:,1,1],K[:,0,2],K[:,1,2] = .58*w,1.92*h,.5*w,.5*h
    data = {}
    sizes = {'MiS':(w,h),'HiS':(int(1.4*w),int(1.4*h)),'LoS':(int(.8*w),int(.8*h))}
    for name,(ww,hh) in sizes.items():
        kk=K.clone();kk[:,0]*=ww/w;kk[:,1]*=hh/h
        data[('K_'+name,0)],data[('inv_K_'+name,0)] = kk,torch.linalg.inv(kk)
        data['dxy_'+name] = torch.tensor([[20.,10.]]*b) if name=='HiS' else torch.zeros(b,2)
        data['resize_'+name] = torch.tensor([[ww,hh]]*b,dtype=torch.float32)
        for f in (0,-1,1):
            data[('color_'+name,f,0)] = torch.rand(b,3,h,w)
    data[('color_MiS_aug',0,0)] = data[('color_MiS',0,0)]
    return data


def core_tests(device):
    torch.manual_seed(123)
    b,h,w=2,32,64
    K=torch.eye(4,device=device).repeat(b,1,1)
    K[:,0,0],K[:,1,1],K[:,0,2],K[:,1,2]=50.,50.,32.,16.
    depth=torch.full((b,1,h,w),3.,device=device,requires_grad=True)
    xy=torch.tensor([[0.,0.],[13.,7.]],device=device)
    image=torch.rand(b,3,h,w,device=device)
    T=torch.eye(4,device=device).repeat(b,1,1)
    out=warp(image,depth,K,torch.linalg.inv(K),T,xy)
    torch.testing.assert_close(out,image,atol=2e-5,rtol=2e-5)
    T[:,0,3]=.01
    warp(image,depth,K,torch.linalg.inv(K),T,xy).mean().backward()
    assert torch.isfinite(depth.grad).all() and depth.grad.abs().sum()>0
    print('PASS: crop-offset projection identity and depth gradient')
    ssim=SSIM().to(device)
    med=torch.full((b,1,h,w),.5,device=device,requires_grad=True)
    hi=torch.full_like(med,.6,requires_grad=True)
    lo=torch.full_like(med,.7,requires_grad=True)
    mask=torch.ones_like(med);mask[0]=0
    hs=torch.tensor([[80,40],[100,50]],device=device)
    ls=torch.tensor([[50,25],[48,24]],device=device)
    xy=torch.tensor([[3,2],[8,5]],device=device)
    ch,cl=consistency(hi,med,lo,mask,hs,ls,xy,ssim)
    lo2=lo.detach().clone();lo2[0]=.01
    ch2,cl2=consistency(hi,med,lo2,mask,hs,ls,xy,ssim)
    torch.testing.assert_close(cl,cl2)
    perm=torch.tensor([1,0],device=device)
    a,c=consistency(hi[perm],med[perm],lo[perm],mask[perm],hs[perm],ls[perm],xy[perm],ssim)
    torch.testing.assert_close(ch,a);torch.testing.assert_close(cl,c)
    (ch+cl).backward()
    for value in (hi,med,lo):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad[0].abs().sum()==0 and value.grad[1].abs().sum()>0
    a,c=consistency(hi,med,lo,torch.zeros_like(mask),hs,ls,xy,ssim)
    assert torch.isfinite(a+c) and float(a+c)==0
    print('PASS: per-sample mask isolation, permutation, bidirectional gradients, empty mask')
    constant=torch.full_like(med,.4)
    a,c=consistency(constant,constant,constant,torch.ones_like(mask),hs,ls,xy,ssim)
    assert float(a+c)<1e-5
    print('PASS: constant predictions remain consistent across all aligned views')
    p=shared_jitter((.8,1.2),(.8,1.2),(.8,1.2),(-.1,.1))
    im=Image.fromarray(np.random.default_rng(1).integers(0,256,(20,30,3),dtype=np.uint8))
    assert np.array_equal(np.asarray(p(im)),np.asarray(p(im)))
    print('PASS: one fixed color transform is shared across temporal frames')


def network_tests(device, integration=False):
    enc=networks.LiteMono(model='lite-mono-tiny',height=192,width=640).to(device)
    dec=networks.TinyDepthDecoder(enc.num_ch_enc,[0]).to(device)
    enc.eval();dec.eval()
    params=sum(p.numel() for m in (enc,dec) for p in m.parameters())
    with torch.no_grad():
        for w,h in [(640,192),(416,128),(512,160),(832,256),(1024,320),(480,144)]:
            out=dec(enc(torch.rand(1,3,h,w,device=device)))[('disp',0)]
            assert out.shape==(1,1,h,w) and torch.isfinite(out).all()
    print('PASS: actual tiny forward at six input sizes; single-head depth parameters',params)
    # Loading a real three-head tiny decoder into the single-head recipe must
    # preserve its full-resolution output, not silently map the wrong weights.
    original=networks.TinyDepthDecoder(enc.num_ch_enc,[0,1,2]).to(device).eval()
    reduced=networks.TinyDepthDecoder(enc.num_ch_enc,[0]).to(device).eval()
    expected=reduced.state_dict()
    reduced.load_state_dict({k:v for k,v in original.state_dict().items() if k in expected},strict=True)
    with torch.no_grad():
        features=enc(torch.rand(1,3,192,640,device=device))
        torch.testing.assert_close(original(features)[('disp',0)],reduced(features)[('disp',0)])
    print('PASS: three-head to single-head checkpoint preserves full-resolution output')
    if integration:
        enc.train();dec.train()
        data={k:v.to(device) for k,v in metadata(1).items()}
        disps={}
        for name in ('HiS','MiS','LoS'):
            key='color_MiS_aug' if name=='MiS' else 'color_'+name
            disps[name]=dec(enc(data[(key,0,0)]))[('disp',0)]
        pose=networks.ResnetEncoder(18,False,num_input_images=2).to(device).train()
        head=networks.PoseDecoder(pose.num_ch_enc,1,2).to(device).train()
        from layers import transformation_from_parameters
        poses={}
        for f in (-1,1):
            pair=[data[('color_MiS',j,0)] for j in ((f,0) if f<0 else (0,f))]
            aa,tr=head([pose(torch.cat(pair,1))])
            poses[f]=transformation_from_parameters(aa[:,0],tr[:,0],invert=(f<0))
        loss=objective(data,disps,poses,SSIM().to(device),'ra')
        loss['loss'].backward()
        for name,model in [('encoder',enc),('decoder',dec),('pose_encoder',pose),('pose_head',head)]:
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert sum(float(g.abs().sum()) for g in grads)>0
        print('PASS: complete three-view tiny + PoseNet + photometric/CS objective backward',float(loss['loss'].detach()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cuda',action='store_true')
    p.add_argument('--integration',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(2)
    device=torch.device('cuda' if a.cuda else 'cpu')
    core_tests(device)
    network_tests(device,a.integration)
    from tiny_ra_eval import eigen_mask, errors
    gt=np.full((100,200),4.)
    assert eigen_mask(gt).any()
    assert errors(gt,np.full_like(gt,4.))[:4]==[0.,0.,0.,0.]
    assert all(x==1. for x in errors(gt,gt)[4:])
    print('PASS: evaluation mask and perfect-prediction metric boundary case')
    print('ALL CHECKS PASSED; synthetic tests do not establish KITTI accuracy.')


if __name__=='__main__':
    main()
