#!/usr/bin/env python3
"""Test large WuBuMath model on DBZ"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

sys.path.insert(0, '/home/wubu/wubucodec/tools')
from wubu_train_large import WuBuCodec

# Load large model checkpoint
ckpt = torch.load('/home/wubu/WuBuMath/output/ckpt_large_e050.pt', map_location='cuda', weights_only=False)
print(f"Loaded checkpoint from epoch {ckpt['epoch']}, loss={ckpt['loss']:.6f}, PSNR={ckpt['psnr']:.1f} dB")

model = WuBuCodec(patch_size=4, latent_dim=32, hidden_dim=256).cuda()
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Load DBZ frame
with open('/tmp/dbz_ab/dbz_input.raw', 'rb') as f:
    frame = f.read(320 * 240 * 3)

frame_np = np.frombuffer(frame, dtype=np.uint8).reshape(240, 320, 3).astype(np.float32) / 255.0
frame_np = frame_np.transpose(2, 0, 1)
frame_tensor = torch.from_numpy(frame_np).unsqueeze(0).cuda()

# Downsample to 64x64
frame_64 = F.interpolate(frame_tensor, size=(64, 64), mode='bilinear', align_corners=False)

# Encode and decode
with torch.no_grad():
    recon_64, latents = model(frame_64)

# Upsample back
recon_240 = F.interpolate(recon_64, size=(240, 320), mode='bilinear', align_corners=False)

# PSNR
mse = F.mse_loss(recon_240, frame_tensor).item()
psnr = 10 * np.log10(1.0 / max(mse, 1e-10))

print(f"\nDBZ frame reconstruction:")
print(f"  MSE: {mse:.6f}")
print(f"  PSNR: {psnr:.1f} dB")
print(f"  Latent shape: {latents.shape}")

# Save reconstruction
recon_np = recon_240[0].permute(1, 2, 0).cpu().numpy()
recon_np = np.clip(recon_np * 255, 0, 255).astype(np.uint8)
with open('/tmp/dbz_ab/recon_wubu_large.raw', 'wb') as f:
    f.write(recon_np.tobytes())
print(f"  Saved to /tmp/dbz_ab/recon_wubu_large.raw")
