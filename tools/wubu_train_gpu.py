#!/usr/bin/env python3
"""
wubu_train_gpu.py — WuBuMath GPU Training on Synthetic Data

Trains encoder-decoder on Poincaré ball with infinite synthetic data.
The model learns the PHYSICS (Poincaré ball, quaternion rotations),
not the content. Random shapes/colors are sufficient for training.

Usage: python3 wubu_train_gpu.py [--epochs 500] [--batch 64] [--data-dir /path/to/synthetic]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import os
import time

# ============================ Poincaré Ball Operations ============================

def poincare_clip(x, c=1.0, eps=1e-6):
    """Clip to Poincaré ball interior"""
    norm = torch.norm(x, dim=-1, keepdim=True)
    max_norm = 1.0 / np.sqrt(c) - eps
    scale = torch.clamp(max_norm / (norm + eps), max=1.0)
    return x * scale

def expmap(v, c=1.0, eps=1e-6):
    """Exponential map: tangent space -> Poincaré ball"""
    norm = torch.norm(v, dim=-1, keepdim=True)
    sqrt_c = np.sqrt(c)
    arg = sqrt_c * norm
    arg = torch.clamp(arg, max=30.0)
    tanh_val = torch.tanh(arg)
    scale = tanh_val / (sqrt_c * norm + eps)
    out = v * scale
    return poincare_clip(out, c, eps)

def logmap(y, c=1.0, eps=1e-6):
    """Logarithmic map: Poincaré ball -> tangent space"""
    y_clipped = poincare_clip(y, c, eps)
    norm = torch.norm(y_clipped, dim=-1, keepdim=True)
    sqrt_c = np.sqrt(c)
    arg = sqrt_c * norm
    arg = torch.clamp(arg, max=1.0 - eps * 4)
    atanh_val = torch.atanh(arg)
    scale = atanh_val / (sqrt_c * norm + eps)
    out = y_clipped * scale
    # Clamp tangent vector norm
    t_norm = torch.norm(out, dim=-1, keepdim=True)
    out = torch.where(t_norm > 10.0, out * 10.0 / (t_norm + eps), out)
    return out

def mobius_add(x, y, c=1.0, eps=1e-6):
    """Möbius addition on Poincaré ball"""
    x_norm_sq = (x * x).sum(dim=-1, keepdim=True)
    y_norm_sq = (y * y).sum(dim=-1, keepdim=True)
    xy_dot = (x * y).sum(dim=-1, keepdim=True)
    denom = 1.0 + 2.0 * c * xy_dot + c * c * x_norm_sq * y_norm_sq
    denom = torch.clamp(denom, min=eps)
    num_x = 1.0 + 2.0 * c * xy_dot + c * y_norm_sq
    num_y = 1.0 - c * x_norm_sq
    out = (num_x * x + num_y * y) / denom
    return poincare_clip(out, c, eps)

def expmap_at(x, v, c=1.0, eps=1e-6):
    """Exponential map at point x: exp_x(v) = x (+) expmap_0(v)"""
    exp_v = expmap(v, c, eps)
    return mobius_add(x, exp_v, c, eps)

def logmap_at(x, y, c=1.0, eps=1e-6):
    """Logarithmic map at point x: log_x(y) = log_0((-x) (+) y)"""
    neg_x = -x
    rel = mobius_add(neg_x, y, c, eps)
    return logmap(rel, c, eps)

def geodesic_interp(x0, x1, t, c=1.0, eps=1e-6):
    """Geodesic interpolation: μ_t = exp_{x0}(t * log_{x0}(x1))"""
    tangent = logmap_at(x0, x1, c, eps)
    tangent_scaled = tangent * t.unsqueeze(-1)
    return expmap_at(x0, tangent_scaled, c, eps)

# ============================ Model ============================

class WuBuEncoder(nn.Module):
    """Encode RGB patches to quaternion latents on Poincaré ball"""
    def __init__(self, patch_size=4, latent_dim=5, hidden_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        input_dim = patch_size * patch_size * 3
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim - 1)  # 4 quaternion components
        )
        
    def forward(self, patches):
        """
        patches: (B, P, patch_size*patch_size*3) - batch of images with P patches
        returns: (B, P, latent_dim) - latents on Poincaré ball
        """
        B, P, _ = patches.shape
        
        # Encode to tangent space
        q_tangent = self.net(patches.reshape(B * P, -1)).reshape(B, P, -1)
        
        # Map to Poincaré ball (quaternion part)
        q_manifold = expmap(q_tangent, c=1.0)
        
        # Amplitude (last dimension): sigmoid to [0,1]
        amp = torch.sigmoid(q_manifold.norm(dim=-1, keepdim=True))
        
        return torch.cat([q_manifold, amp], dim=-1)

class WuBuDecoder(nn.Module):
    """Decode quaternion latents back to RGB patches"""
    def __init__(self, patch_size=4, latent_dim=5, hidden_dim=64):
        super().__init__()
        self.patch_size = patch_size
        output_dim = patch_size * patch_size * 3
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
    def forward(self, latents):
        """
        latents: (B, P, latent_dim) - latents on Poincaré ball
        returns: (B, P, patch_size*patch_size*3) - reconstructed patches
        """
        return self.net(latents)

class WuBuCodec(nn.Module):
    """Full codec: encoder + decoder"""
    def __init__(self, patch_size=4, latent_dim=5, hidden_dim=64):
        super().__init__()
        self.encoder = WuBuEncoder(patch_size, latent_dim, hidden_dim)
        self.decoder = WuBuDecoder(patch_size, latent_dim, hidden_dim)
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        
    def forward(self, images):
        """
        images: (B, C, H, W) - batch of RGB images
        returns: (B, C, H, W) - reconstructed images
        """
        B, C, H, W = images.shape
        PS = self.patch_size
        
        # Extract patches
        patches = images.unfold(2, PS, PS).unfold(3, PS, PS)  # (B, C, PH, PW, PS, PS)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, PH, PW, C, PS, PS)
        PH, PW = patches.shape[1], patches.shape[2]
        patches = patches.reshape(B, PH * PW, C * PS * PS)  # (B, P, PS*PS*C)
        
        # Encode to latent
        latents = self.encoder(patches)  # (B, P, latent_dim)
        
        # Decode
        recon_patches = self.decoder(latents)  # (B, P, PS*PS*C)
        
        # Reconstruct image
        recon_patches = recon_patches.reshape(B, PH, PW, C, PS, PS)
        recon_patches = recon_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        recon = recon_patches.reshape(B, C, H, W)
        
        return recon, latents

# ============================ Synthetic Data Generator ============================

def generate_synthetic_batch(batch_size, img_size=64, patch_size=4, device=None):
    """Generate a batch of synthetic images with random shapes and colors"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PS = patch_size
    H = W = img_size
    
    images = torch.zeros(batch_size, 3, H, W, device=device)
    
    for b in range(batch_size):
        # Background: Poincaré ball random walk
        bg_val = torch.rand(3, device=device)
        images[b] = bg_val.view(3, 1, 1).expand(3, H, W)
        
        # Random circles
        num_circles = np.random.randint(3, 8)
        for c in range(num_circles):
            cx = torch.rand(1, device=device).item() * W
            cy = torch.rand(1, device=device).item() * H
            radius = np.random.uniform(5, 20)
            color = torch.rand(3, device=device)
            
            y_coords = torch.arange(H, device=device).view(-1, 1).float()
            x_coords = torch.arange(W, device=device).view(1, -1).float()
            dist = ((x_coords - cx)**2 + (y_coords - cy)**2).sqrt()
            mask = (dist < radius).float()
            
            for ch in range(3):
                images[b, ch] = images[b, ch] * (1 - mask) + color[ch] * mask
        
        # Random rectangles
        num_rects = np.random.randint(2, 5)
        for r in range(num_rects):
            rx = torch.rand(1, device=device).item() * W
            ry = torch.rand(1, device=device).item() * H
            rw = np.random.uniform(10, 30)
            rh = np.random.uniform(10, 30)
            color = torch.rand(3, device=device)
            
            y_coords = torch.arange(H, device=device).view(-1, 1).float()
            x_coords = torch.arange(W, device=device).view(1, -1).float()
            mask = ((x_coords >= rx) & (x_coords < rx + rw) & 
                    (y_coords >= ry) & (y_coords < ry + rh)).float()
            
            for ch in range(3):
                images[b, ch] = images[b, ch] * (1 - mask) + color[ch] * mask
    
    return images

# ============================ Training ============================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = WuBuCodec(patch_size=4, latent_dim=5, hidden_dim=args.hidden).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'train_log_gpu.csv')
    log_file = open(log_path, 'w')
    log_file.write('epoch,loss,psnr,lr\n')
    
    print(f"\nTraining: {args.epochs} epochs, batch {args.batch}")
    print(f"Image: {args.img_size}x{args.img_size}, patches: 256, latent: 5")
    print(f"Compression: {args.img_size*args.img_size*3*4:.0f}:1\n")
    
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        num_batches = args.batches_per_epoch
        
        for batch in range(num_batches):
            # Generate synthetic data
            images = generate_synthetic_batch(args.batch, args.img_size, 4, device)
            
            # Forward pass
            recon, latents = model(images)
            
            # Loss: MSE + Poincaré ball constraint (keep latents on manifold)
            mse_loss = F.mse_loss(recon, images)
            
            # Poincaré constraint: quaternion part should have norm < 1
            q = latents[:, :, :4]
            q_norm = torch.norm(q, dim=-1)
            poincare_loss = F.relu(q_norm - 0.999).mean()
            
            # Amplitude should be in [0,1] (handled by sigmoid in encoder)
            amp = latents[:, :, 4]
            amp_loss = F.relu(-amp).mean() + F.relu(amp - 1.0).mean()
            
            loss = mse_loss + 0.1 * poincare_loss + 0.1 * amp_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        epoch_loss /= num_batches
        
        # Compute PSNR
        with torch.no_grad():
            model.eval()
            test_images = generate_synthetic_batch(16, args.img_size, 4, device)
            test_recon, _ = model(test_images)
            test_mse = F.mse_loss(test_recon, test_images).item()
            psnr = 10 * np.log10(1.0 / max(test_mse, 1e-10))
        
        lr = optimizer.param_groups[0]['lr']
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {epoch_loss:.6f} | PSNR: {psnr:.1f} dB | LR: {lr:.6f} | Time: {elapsed:.1f}s")
            log_file.write(f"{epoch+1},{epoch_loss:.6f},{psnr:.2f},{lr:.6f}\n")
        
        # Save checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            ckpt_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'psnr': psnr,
            }, ckpt_path)
        
        if (epoch + 1) % 50 == 0:
            ckpt_path = os.path.join(args.output_dir, f'ckpt_e{epoch+1:03d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'psnr': psnr,
            }, ckpt_path)
    
    log_file.close()
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'wubu_gpu_final.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'loss': epoch_loss,
        'psnr': psnr,
    }, final_path)
    
    print(f"\nDone. Best loss: {best_loss:.6f}")
    print(f"Final PSNR: {psnr:.1f} dB")
    print(f"Model saved to: {final_path}")
    print(f"Log saved to: {log_path}")
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WuBuMath GPU Training')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--img-size', type=int, default=64)
    parser.add_argument('--batches-per-epoch', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default='/home/wubu/WuBuMath/output')
    args = parser.parse_args()
    
    train(args)
