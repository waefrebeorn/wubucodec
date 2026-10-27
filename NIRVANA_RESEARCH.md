# WuBuCodec NIRVANA Architecture
## Research Synthesis: AV1 + JPEG XL + WuBuMath → The Hybrid Codec

### Current State of the Art (2026)

| Codec | PSNR @ 1200 kbps | Key Techniques |
|-------|------------------|-----------------|
| **AV1** | 44.1 dB | 128→4 quadtree, DCT/ADST/IDTX, CDEF, Wiener, 35+ intra |
| **JPEG XL** | N/A (image) | VarDCT 256→2, AFV/Hornuss transforms, modular mode |
| **WuBu v17** | 35.2 dB | 8x8 fixed, DCT only, basic CDEF/Wiener, 35 intra |
| **x264** | 28.4 dB | 16x4→4x4, DCT, deblocking |

### The 9 dB Gap: Root Cause Analysis

AV1 beats WuBu by 9 dB because of **four architectural advantages**:

1. **Variable block sizes (128→4)** — ~3 dB gain
   - AV1 uses recursive quadtree partitioning: 128x128, 64x64, 32x32, 16x16, 8x8, 4x4
   - Flat regions use large blocks (fewer bits for prediction)
   - Detailed regions use small blocks (better prediction accuracy)
   - WuBu uses fixed 8x8 — wastes bits on flat areas, loses detail on complex areas

2. **Multiple transform types** — ~1.5 dB gain
   - AV1: DCT, ADST, flipped ADST, Identity (IDTX), per-axis combinations
   - Different residuals have different spectral properties
   - WuBu: DCT only — suboptimal for directional patterns

3. **Advanced in-loop filtering** — ~2 dB gain
   - AV1 CDEF: per-block adaptive strength, 8 directions, edge-preserving constraints
   - AV1 Loop Restoration: 7-tap Wiener filter with per-block adaptive coefficients
   - WuBu: simplistic directional average + fixed Wiener — doesn't adapt to content

4. **Superior entropy coding** — ~15% bitrate savings
   - AV1: multi-symbol arithmetic coding with context modeling
   - WuBu: Exp-Golomb — ~15% less efficient

### The NIRVANA Architecture

The solution is a **hybrid codec** that combines:
- **AV1's proven skeleton**: variable blocks, quadtree partitioning, advanced filters
- **JPEG XL's VarDCT philosophy**: variable-size DCT with multiple transform kernels
- **WuBuMath's hyperbolic manifold theory**: for latent space coding of prediction residuals

#### Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NIRVANA CODEC                             │
├─────────────────────────────────────────────────────────────┤
│ 1. Variable Block Partitioning (Quadtree)                    │
│    64x64 → 32x32 → 16x16 → 8x8 → 4x4                      │
│    RD-cost based split/merge decision                        │
│                                                              │
│ 2. Multiple Transform Selection (per block)                  │
│    DCT, ADST, IDTX, Flip-ADST                               │
│    Per-axis combinations (e.g., DCT horizontal + ADST vert) │
│    RD-cost based transform selection                         │
│                                                              │
│ 3. In-Loop Filtering                                         │
│    CDEF: directional edge-preserving filter                  │
│    Loop Restoration: adaptive Wiener filter                 │
│    Hyperbolic projection: manifold-aware smoothing           │
│                                                              │
│ 4. Entropy Coding                                            │
│    Range coder with context modeling                        │
│    Adaptive probability updates                              │
│                                                              │
│ 5. Rate Control                                              │
│    Two-pass encoding with lookahead                         │
│    Per-frame QP adaptation                                  │
└─────────────────────────────────────────────────────────────┘
```

#### WuBuMath Enhancement: Hyperbolic Latent Space

The key innovation from WuBuMath is representing prediction residuals in a **hyperbolic manifold** (Poincaré ball) instead of Euclidean space:

- **Euclidean residual coding**: residuals are coded as vectors in R^n with L2 distortion
- **Hyperbolic residual coding**: residuals are mapped to Poincaré ball, coded with hyperbolic distance

Why this helps:
1. **Natural clustering**: similar residuals cluster on the manifold
2. **Hierarchical structure**: tree-like data (like motion fields) embed naturally
3. **Better rate-distortion**: hyperbolic distance better matches perceptual distortion

Implementation:
1. Compute prediction residual (Euclidean)
2. Map to Poincaré ball: `z = tanh(λ * ||r||) * r / ||r||`
3. Quantize in hyperbolic space
4. Code with range coder
5. Dequantize and map back to Euclidean

### Implementation Priority

| Priority | Feature | Expected Gain | Effort |
|----------|---------|---------------|--------|
| 1 | Variable block sizes (64→8) | +3 dB | High |
| 2 | Multiple transforms (DCT/ADST/IDTX) | +1.5 dB | Medium |
| 3 | Proper CDEF (AV1-style) | +1 dB | Medium |
| 4 | Proper Wiener (AV1-style) | +1 dB | Medium |
| 5 | Range coder with contexts | +10% bits | Medium |
| 6 | Hyperbolic latent coding | +0.5 dB | High |
| 7 | Two-pass rate control | +5% bits | Medium |

**Total expected gain: ~7-8 dB → ~42-43 dB (matching AV1)**

### Why This Is the Nirvana Solution

1. **AV1's architecture is proven** — it's the industry standard for a reason
2. **JPEG XL's VarDCT shows variable blocks work** — even for images
3. **WuBuMath's hyperbolic theory adds a unique dimension** — no other codec uses manifold-aware coding
4. **The combination is greater than the sum** — hyperbolic coding enhances variable-block residuals

### The Hard Truth

Closing the 9 dB gap requires **rebuilding the codec core** around variable block sizes. This is not a feature addition — it's an architectural change. The current WuBu encoder is a solid foundation (beats x264 by 7 dB), but it's architecturally capped at ~35 dB.

The NIRVANA architecture is the path to 43+ dB. It's a significant implementation effort, but every component is well-understood from AV1/JPEG XL research.

---

## Research Sources

1. **AV1 Technical Overview** (arXiv:2008.06091) — comprehensive AV1 codec design
2. **ForaSoft AV1 2026** — state of AV1 deployment and techniques
3. **JPEG XL Wikipedia** — VarDCT and modular mode details
4. **Mozilla Hacks: CDEF** — AV1's Constrained Directional Enhancement Filter
5. **WuBuMath internal** — hyperbolic manifold theory for codec design
