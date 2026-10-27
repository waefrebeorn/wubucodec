#!/usr/bin/env python3
"""Test WuBuMath trained model on DBZ frame"""
import torch
import torch.nn as nn
import numpy as np
import sys

# Add tools path
sys.path.insert(0, '/home/wubu/wubucodec/tools')

# Import the model from training script
from wubu_train_gpu import WuBuCodec, generate_synthetic_batch

# Load checkpoint
ckpt = torch.load('/home/wubu/WuBuMath/output/best_model.pt', map_location='cuda', weights_only=False)
print(f"Loaded checkpoint from epoch {ckpt['epoch']}, loss={ckpt['loss']:.6f}, PSNR={ckpt['psnr']:.1f} dB")

# Create model and load weights
model = WuBuCodec(patch_size=4, latent_dim=5, hidden_dim=64).cuda()
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Load DBZ frame
import struct
with open('/tmp/dbz_ab/dbz_input.raw', 'rb') as f:
    frame = f.read(320 * 240 * 3)

# Convert to tensor
frame_np = np.frombuffer(frame, dtype=np.uint8).reshape(240, 320, 3).astype(np.float32) / 255.0
frame_np = frame_np.transpose(2, 0, 1)  # CHW
frame_tensor = torch.from_numpy(frame_np).unsqueeze(0).cuda()  # 1CHW

# Downsample to 64x64
import torch.nn.functional as F
frame_64 = F.interpolate(frame_tensor, size=(64, 64), mode='bilinear', align_corners=False)

# Encode and decode
with torch.no_grad():
    recon_64, latents = model(frame_64)

# Upsample back to 240x320
recon_240 = F.interpolate(recon_64, size=(240, 320), mode='bilinear', align_corners=False)

# Compute PSNR
mse = F.mse_loss(recon_240, frame_tensor).item()
psnr = 10 * np.log10(1.0 / max(mse, 1e-10))

print(f"\nDBZ frame reconstruction:")
print(f"  MSE: {mse:.6f}")
print(f"  PSNR: {psnr:.1f} dB")

# Latent stats
print(f"\nLatent stats:")
print(f"  Shape: {latents.shape}")
print(f"  Min: {latents.min().item():.3f}")
print(f"  Max: {latents.max().item():.3f}")
print(f"  Mean: {latents.mean().item():.3f}")
print(f"  Std: {latents.std().item():.3f}")

# Quaternion norms
q = latents[0, :, :4]
q_norms = torch.norm(q, dim=-1)
print(f"\nQuaternion norms:")
print(f"  Min: {q_norms.min().item():.3f}")
print(f"  Max: {q_norms.max().item():.3f}")
print(f"  Mean: {q_norms.mean().item():.3f}")

# Save reconstruction as raw
recon_np = recon_240[0].permute(1, 2, 0).cpu().numpy()  # HWC
recon_np = np.clip(recon_np * 255, 0, 255).astype(np.uint8)
with open('/tmp/dbz_ab/recon_wubu.raw', 'wb') as f:
    f.write(recon_np.tobytes())

print(f"\nReconstruction saved to /tmp/dbz_ab/recon_wubu.raw")

# Test on synthetic data too
syn_images = generate_synthetic_batch(16, 64, 4, 'cuda')
with torch.no_grad():
    syn_recon, syn_latents = model(syn_images)
syn_mse = F.mse_loss(syn_recon, syn_images).item()
syn_psnr = 10 * np.log10(1.0 / max(syn_mse, 1e-10))
print(f"\nSynthetic test PSNR: {syn_psnr:.1f} dB")
