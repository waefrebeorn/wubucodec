#!/usr/bin/env python3
"""
wubu_train_real_video.py — Train WuBuMath model directly on real DBZ video

Instead of synthetic data, use actual DBZ frames for training.
This should bridge the generalization gap.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time

# Poincaré ball operations
def poincare_clip(x, c=1.0, eps=1e-6):
    norm = torch.norm(x, dim=-1, keepdim=True)
    max_norm = 1.0 / np.sqrt(c) - eps
    scale = torch.clamp(max_norm / (norm + eps), max=1.0)
    return x * scale

def expmap(v, c=1.0, eps=1e-6):
    norm = torch.norm(v, dim=-1, keepdim=True)
    sqrt_c = np.sqrt(c)
    arg = torch.clamp(sqrt_c * norm, max=30.0)
    scale = torch.tanh(arg) / (sqrt_c * norm + eps)
    return poincare_clip(v * scale, c, eps)

# Model
class WuBuEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(48, 128), nn.GELU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 31)
        )
    def forward(self, patches):
        B, P, _ = patches.shape
        q_tangent = self.net(patches.reshape(B*P, -1)).reshape(B, P, -1)
        q_manifold = expmap(q_tangent)
        amp = torch.sigmoid(q_manifold.norm(dim=-1, keepdim=True))
        return torch.cat([q_manifold, amp], dim=-1)

class WuBuDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, 48), nn.Sigmoid()
        )
    def forward(self, latents):
        return self.net(latents)

class WuBuCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = WuBuEncoder()
        self.decoder = WuBuDecoder()
    def forward(self, images):
        B, C, H, W = images.shape
        PS = 4
        patches = images.unfold(2, PS, PS).unfold(3, PS, PS)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        PH, PW = patches.shape[1], patches.shape[2]
        patches = patches.reshape(B, PH*PW, C*PS*PS)
        latents = self.encoder(patches)
        recon_patches = self.decoder(latents)
        recon_patches = recon_patches.reshape(B, PH, PW, C, PS, PS)
        recon_patches = recon_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        recon = recon_patches.reshape(B, C, H, W)
        return recon, latents

# Load real video frames
def load_real_frames(path, W=320, H=240, max_frames=512):
    with open(path, 'rb') as f:
        data = f.read()
    nf = len(data) // (W * H * 3)
    nf = min(nf, max_frames)
    frames = np.zeros((nf, 3, 64, 64), dtype=np.float32)
    for i in range(nf):
        frame = np.frombuffer(data[i*W*H*3:(i+1)*W*H*3], dtype=np.uint8).reshape(H, W, 3)
        # Downsample to 64x64
        for y in range(64):
            for x in range(64):
                sy, sx = y * H // 64, x * W // 64
                frames[i, :, y, x] = frame[sy, sx] / 255.0
    return frames

def train():
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # Load real video
    frames = load_real_frames('/tmp/dbz_ab/dbz_input.raw')
    print(f"Loaded {len(frames)} real frames")
    
    model = WuBuCodec().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    data = torch.from_numpy(frames).to(device)
    
    os.makedirs('/home/wubu/WuBuMath/output', exist_ok=True)
    log_file = open('/home/wubu/WuBuMath/output/train_log_real_video.csv', 'w')
    log_file.write('epoch,loss,psnr\n')
    
    best_loss = float('inf')
    start = time.time()
    
    for epoch in range(200):
        model.train()
        # Mini-batch training
        indices = torch.randperm(len(data))
        epoch_loss = 0
        num_batches = len(data) // 16
        
        for b in range(num_batches):
            idx = indices[b*16:(b+1)*16]
            batch = data[idx]
            
            recon, latents = model(batch)
            loss = F.mse_loss(recon, batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        epoch_loss /= num_batches
        
        if (epoch + 1) % 10 == 0:
            psnr = 10 * np.log10(1.0 / max(epoch_loss, 1e-10))
            elapsed = time.time() - start
            print(f"Epoch {epoch+1:3d}/200 | Loss: {epoch_loss:.6f} | PSNR: {psnr:.1f} dB | Time: {elapsed:.1f}s")
            log_file.write(f"{epoch+1},{epoch_loss:.6f},{psnr:.2f}\n")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': epoch_loss,
            }, '/home/wubu/WuBuMath/output/best_real_video.pt')
    
    log_file.close()
    print(f"\nDone. Best loss: {best_loss:.6f}")

if __name__ == '__main__':
    train()
