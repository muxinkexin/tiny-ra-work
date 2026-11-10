"""Original implementation of the RA training objective; no architecture changes."""
import torch
import torch.nn.functional as F


def shared_jitter(brightness, contrast, saturation, hue):
    from torchvision.transforms import ColorJitter, functional as TF
    order, b, c, s, h = ColorJitter.get_params(brightness, contrast, saturation, hue)
    def apply(image):
        for op in order.tolist():
            if op == 0 and b is not None:
                image = TF.adjust_brightness(image, b)
            elif op == 1 and c is not None:
                image = TF.adjust_contrast(image, c)
            elif op == 2 and s is not None:
                image = TF.adjust_saturation(image, s)
            elif op == 3 and h is not None:
                image = TF.adjust_hue(image, h)
        return image
    return apply


def warp(source, depth, K, inv_K, T, dxy):
    """RA crop-offset convention: unprojection adds dxy; projection subtracts it.

    K is in the resized, UN-cropped image coordinates. Do not subtract dxy
    from K a second time. Low-scale stitching follows the upstream convention.
    """
    b, _, h, w = depth.shape
    if h < 2 or w < 2:
        raise ValueError('Projection requires height and width >= 2')
    yy, xx = torch.meshgrid(torch.arange(h, device=depth.device, dtype=depth.dtype),
                            torch.arange(w, device=depth.device, dtype=depth.dtype), indexing='ij')
    pixels = torch.stack([xx, yy], 0).reshape(1, 2, -1).expand(b, -1, -1)
    pixels = pixels + dxy.to(depth).unsqueeze(-1)
    ones = torch.ones(b, 1, h*w, device=depth.device, dtype=depth.dtype)
    pixels = torch.cat([pixels, ones], 1)
    xyz = (inv_K[:, :3, :3] @ pixels) * depth.reshape(b, 1, -1)
    points = torch.cat([xyz, ones], 1)
    projected = ((K @ T)[:, :3, :] @ points)
    uv = projected[:, :2] / (projected[:, 2:3] + 1e-7)
    uv = uv - dxy.to(depth).unsqueeze(-1)
    gx = 2 * uv[:, 0] / (w-1) - 1
    gy = 2 * uv[:, 1] / (h-1) - 1
    grid = torch.stack([gx, gy], -1).reshape(b, h, w, 2)
    return F.grid_sample(source, grid, padding_mode='border', align_corners=True)


def photometric(pred, target, ssim):
    return .85 * ssim(pred, target).mean(1, keepdim=True) + .15 * (pred-target).abs().mean(1, keepdim=True)


def reconstruction(target, sources, predictions, ssim):
    reproj = torch.cat([photometric(p, target, ssim) for p in predictions], 1)
    identity = torch.cat([photometric(s, target, ssim) for s in sources], 1)
    identity = identity + torch.randn_like(identity) * 1e-5
    values, index = torch.cat([identity, reproj], 1).min(1, keepdim=True)
    valid = (index >= len(sources)).to(target.dtype).detach()
    return values.mean(), valid


def masked_mean(values, mask):
    if values.shape != mask.shape:
        raise ValueError(f'Loss/mask shape mismatch: {values.shape} != {mask.shape}')
    return (values * mask).sum() / (mask.sum() + 1e-7)


def consistency(high, medium, low, mask, high_size, low_size, high_xy, ssim):
    """Upstream high-view rounded crop alignment, corrected per-item low mask.

    Operates on sigmoid disparity, with gradients into BOTH sides. This is
    intentionally not a log-depth loss and not teacher/student distillation.
    """
    b, _, h, w = medium.shape
    high_losses, low_losses = [], []
    for i in range(b):
        wh, hh = [int(v) for v in high_size[i].tolist()]
        wl, hl = [int(v) for v in low_size[i].tolist()]
        dx, dy = [int(v) for v in high_xy[i].tolist()]
        x, y = round(w * dx / wh), round(h * dy / hh)
        cw, ch = round(w*w/wh), round(h*h/hh)
        if not (0 <= x <= w-cw and 0 <= y <= h-ch and cw >= 3 and ch >= 3):
            raise ValueError(f'Invalid high-view overlap: {(x,y,cw,ch)}')
        high_aligned = F.interpolate(high[i:i+1], (ch, cw), mode='bilinear', align_corners=False)
        med_roi = medium[i:i+1, :, y:y+ch, x:x+cw]
        m_roi = mask[i:i+1, :, y:y+ch, x:x+cw]
        high_losses.append(masked_mean(photometric(med_roi, high_aligned, ssim), m_roi))
        low_roi = low[i:i+1, :, :hl, :wl]
        low_aligned = F.interpolate(low_roi, (h, w), mode='bilinear', align_corners=False)
        low_losses.append(masked_mean(photometric(medium[i:i+1], low_aligned, ssim), mask[i:i+1]))
    return torch.stack(high_losses).mean(), torch.stack(low_losses).mean()


def objective(inputs, disparities, poses, ssim, mode, beta=1., min_depth=.1, max_depth=100., smooth_weight=1e-3):
    from layers import disp_to_depth, get_smooth_loss
    names = ['MiS'] if mode == 'base' else ['HiS', 'MiS', 'LoS']
    losses, masks = {}, {}
    for name in names:
        source_name = 'MiS' if mode == 'repeat' else name
        disp = disparities[name]
        _, depth = disp_to_depth(disp, min_depth, max_depth)
        K, inv_K = inputs[('K_'+source_name, 0)], inputs[('inv_K_'+source_name, 0)]
        dxy = inputs['dxy_'+source_name]
        target = inputs[('color_'+source_name, 0, 0)]
        sources = [inputs[('color_'+source_name, f, 0)] for f in (-1, 1)]
        preds = [warp(src, depth, K, inv_K, poses[f], dxy) for src, f in zip(sources, (-1, 1))]
        photo, mask = reconstruction(target, sources, preds, ssim)
        smooth = get_smooth_loss(disp/(disp.mean((2,3), keepdim=True)+1e-7), target)
        losses['photo_'+name], losses['smooth_'+name] = photo, smooth
        losses['branch_'+name] = photo + smooth_weight * smooth
        masks[name] = mask
    zero = disparities['MiS'].sum() * 0
    ch, cl = zero, zero
    if mode == 'ra':
        ch, cl = consistency(disparities['HiS'], disparities['MiS'], disparities['LoS'], masks['MiS'],
                             inputs['resize_HiS'], inputs['resize_LoS'], inputs['dxy_HiS'], ssim)
    losses['cs_high'], losses['cs_low'] = ch, cl
    losses['mask_coverage'] = masks['MiS'].mean()
    losses['loss'] = sum(losses['branch_'+n] for n in names) + beta*(ch+cl)
    return losses
