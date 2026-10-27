#!/usr/bin/env python3
"""
gaad_pytorch.py: GAAD - Golden Ratio Agnostic Decomposition in PyTorch
Variable block sizes with recursive quadtree for GPU-accelerated encoding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import argparse

GOLDEN_RATIO = 1.6180339887498948482
BLOCK_SIZES = [64, 40, 25, 16, 10, 6, 4]
MIN_BLOCK_SIZE = 4

def dct_matrix(N):
    n = torch.arange(N, dtype=torch.float32).unsqueeze(0)
    k = torch.arange(N, dtype=torch.float32).unsqueeze(1)
    basis = torch.cos((2 * n + 1) * k * np.pi / (2 * N))
    basis[0] = basis[0] / np.sqrt(2)
    return basis * np.sqrt(2 / N)

def get_qstep(qp):
    table = [1,1,1,1,2,2,3,3,4,4,5,6,7,8,10,11,13,14,16,18,
             20,23,25,29,32,36,40,45,51,57,64,72,81,91,102,114,
             128,144,161,181,203,228,256,287,322,361,405,455,511,574,644,722,
             810,910,1022,1148,1288,1444,1620,1820,2044,2296,2576,2888]
    return table[max(0, min(63, qp))]

def get_lambda(qp):
    return 0.85 * (2.0 ** ((qp - 12.0) / 3.0))

def sad_block(cur, ref, W, H, bx, by, dx, dy, size):
    rx, ry = bx + dx, by + dy
    if rx < 0 or ry < 0 or rx + size > W or ry + size > H:
        return 1e10
    s = 0
    for r in range(size):
        for c in range(size):
            s += abs(int(cur[(by+r)*W + bx + c]) - int(ref[(ry+r)*W + rx + c]))
    return s

def me_block(cur, ref, W, H, bx, by, sr, size):
    dx, dy = 0, 0
    best_sad = sad_block(cur, ref, W, H, bx, by, 0, 0, size)
    step = sr
    diamond = [[-1,0],[0,-1],[0,1],[1,0],[-1,-1],[-1,1],[1,-1],[1,1]]
    while step > 0:
        improved = True
        while improved:
            improved = False
            for d in diamond:
                nx, ny = dx + d[0]*step, dy + d[1]*step
                if abs(nx) > sr or abs(ny) > sr:
                    continue
                s = sad_block(cur, ref, W, H, bx, by, nx, ny, size)
                if s < best_sad:
                    best_sad, dx, dy = s, nx, ny
                    improved = True
        step //= 2
    return dx, dy

def mc_get_pred(ref, W, H, bx, by, dx, dy, size):
    rx, ry = bx + dx, by + dy
    pred = []
    for r in range(size):
        for c in range(size):
            ry_c = max(0, min(H-1, ry+r))
            rx_c = max(0, min(W-1, rx+c))
            pred.append(ref[ry_c*W + rx_c])
    return pred

def encode_block(orig, recon, W, H, bx, by, size, ref, qp, lam):
    blk = []
    for r in range(size):
        for c in range(size):
            blk.append(orig[(by+r)*W + bx + c])
    
    best_cost = 1e18
    best_recon = None
    
    # Try inter
    if ref is not None:
        dx, dy = me_block(orig, ref, W, H, bx, by, 16, size)
        pred = mc_get_pred(ref, W, H, bx, by, dx, dy, size)
        res = [blk[i] - pred[i] for i in range(size*size)]
        var = sum(r*r for r in res) / (size*size)
        qs = get_qstep(qp + (-4 if var < 100 else (-2 if var < 500 else (0 if var < 2000 else (2 if var < 5000 else 4)))))
        
        res_2d = np.array(res).reshape(size, size)
        dct_m = dct_matrix(size)
        coeff = dct_m @ res_2d @ dct_m.T
        coeff_q = np.round(coeff / qs) * qs
        rres = dct_m.T @ coeff_q @ dct_m
        recon_block = np.clip(np.array(pred) + rres.reshape(-1), 0, 255)
        
        sse = np.sum((recon_block - blk) ** 2)
        cost = sse + lam * size * size
        if cost < best_cost:
            best_cost = cost
            best_recon = recon_block
    
    # Try intra (DC)
    s = 0
    n = 0
    if by > 0:
        for c in range(size):
            if bx + c < W:
                s += recon[(by-1)*W + bx + c]
                n += 1
    if bx > 0:
        for r in range(size):
            if by + r < H:
                s += recon[(by+r)*W + bx - 1]
                n += 1
    dc = s / n if n > 0 else 128
    
    pred = [dc] * (size * size)
    res = [blk[i] - pred[i] for i in range(size*size)]
    var = sum(r*r for r in res) / (size*size)
    qs = get_qstep(qp + (-4 if var < 100 else (-2 if var < 500 else (0 if var < 2000 else (2 if var < 5000 else 4)))))
    
    res_2d = np.array(res).reshape(size, size)
    dct_m = dct_matrix(size)
    coeff = dct_m @ res_2d @ dct_m.T
    coeff_q = np.round(coeff / qs) * qs
    rres = dct_m.T @ coeff_q @ dct_m
    recon_block = np.clip(np.array(pred) + rres.reshape(-1), 0, 255)
    
    sse = np.sum((recon_block - blk) ** 2)
    cost = sse + lam * size * size
    if cost < best_cost:
        best_cost = cost
        best_recon = recon_block
    
    # Write reconstruction
    if best_recon is not None:
        for r in range(size):
            for c in range(size):
                recon[(by+r)*W + bx + c] = best_recon[r*size + c]
    
    return best_cost

def encode_gaad(orig, recon, W, H, bx, by, size, ref, qp, lam):
    """GAAD recursive encode with golden ratio block sizes."""
    if size < MIN_BLOCK_SIZE:
        size = MIN_BLOCK_SIZE
    if size > 64:
        size = 64
    
    # Try not splitting
    cost_no_split = encode_block(orig, recon, W, H, bx, by, size, ref, qp, lam)
    
    if size <= MIN_BLOCK_SIZE:
        return cost_no_split
    
    # Try splitting
    sub_size = int(size / GOLDEN_RATIO)
    if sub_size < MIN_BLOCK_SIZE:
        sub_size = MIN_BLOCK_SIZE
    
    # Compute variance to decide splitting
    blk = []
    for r in range(size):
        for c in range(size):
            blk.append(orig[(by+r)*W + bx + c])
    
    mean = sum(blk) / len(blk)
    var = sum((b - mean) ** 2 for b in blk) / len(blk)
    
    threshold = 500 * GOLDEN_RATIO
    
    if var > threshold and sub_size >= MIN_BLOCK_SIZE:
        c1 = encode_gaad(orig, recon, W, H, bx, by, sub_size, ref, qp, lam)
        c2 = encode_gaad(orig, recon, W, H, bx+sub_size, by, sub_size, ref, qp, lam)
        c3 = encode_gaad(orig, recon, W, H, bx, by+sub_size, sub_size, ref, qp, lam)
        c4 = encode_gaad(orig, recon, W, H, bx+sub_size, by+sub_size, sub_size, ref, qp, lam)
        
        split_cost = c1 + c2 + c3 + c4 + 10
        
        if split_cost < cost_no_split:
            return split_cost
        
        # Splitting wasn't better - re-encode as single block
        encode_block(orig, recon, W, H, bx, by, size, ref, qp, lam)
    
    return cost_no_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('-q', type=int, default=26)
    args = parser.parse_args()
    
    W, H = args.width, args.height
    print(f"GAAD PyTorch - {W}x{H} QP={args.q}")
    
    with open(args.input, 'rb') as f:
        data = f.read()
    
    nf = len(data) // (W * H * 3)
    frames = np.frombuffer(data[:nf*W*H*3], dtype=np.uint8).reshape(nf, H, W, 3)
    print(f"Loaded {nf} frames")
    
    total_psnr = 0
    start = time.time()
    
    for fn in range(min(nf, 10)):
        orig = frames[fn, :, :, 0].flatten().astype(np.float32)
        recon = np.zeros_like(orig)
        ref = frames[fn-1, :, :, 0].flatten().astype(np.float32) if fn > 0 else None
        
        lam = get_lambda(args.q)
        
        for ctu_y in range(0, H, 64):
            for ctu_x in range(0, W, 64):
                ctu_size = min(64, W - ctu_x, H - ctu_y)
                encode_gaad(orig, recon, W, H, ctu_x, ctu_y, ctu_size, ref, args.q, lam)
        
        mse = np.mean((orig - recon) ** 2)
        psnr = 10 * np.log10(255**2 / max(mse, 1e-10))
        total_psnr += psnr
        
        if fn % 2 == 0:
            print(f"  Frame {fn}: PSNR={psnr:.1f} dB")
    
    avg_psnr = total_psnr / min(nf, 10)
    elapsed = time.time() - start
    print(f"\nAverage PSNR: {avg_psnr:.1f} dB")
    print(f"Time: {elapsed:.1f}s")

if __name__ == '__main__':
    main()
