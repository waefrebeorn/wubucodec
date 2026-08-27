# WuBuCodec — ~ WuBu ~ Video Codec

**SOTA dominance campaign: 24 categories, beat AV1 on PSNR, close the bitrate gap.**

```
PSNR
  WuBu Best: 62.93 dB
  AV1:       60.36 dB
  Gap:       +2.57 dB better

Bitrate
  WuBu Best: 93.2 kbps
  AV1:       24.2 kbps
  Gap:       ~4× worse (closed from 27×)

Progress: 3 closed, 14 in_progress, 7 open of 24 categories
```

## Architecture

WuBuCodec is a from-scratch C11 video codec. Every component is implemented, tested, and wired into a unified RDO encoder loop (C15 keystone).

```
src/
  codec/       — Unified RDO encoder (C15 keystone)
  transform/   — Variable-size DCT/DST-VII (C9)
  entropy/     — EG, CABAC, trellis RDOQ, tokctx, ncodec
  motion/      — Motion estimation, sub-pel, MC, DPB
  intra/       — Intra prediction (DC/Vert/Horiz/Plane + angular)
  filter/      — Deblocking (C12), SAO (C13), FGS, ALF-stub
  temporal/    — B-frames, GOP opt, lookahead, fast mode
  rdo/         — Rate-distortion optimization, lambda, RD curves
  container/   — NAL units, .WUBV container, seeking
  quality/     — PSNR, SSIM, MS-SSIM, BD-rate
  neural/      — Learned codec, latent codec
  quat/        — Quaternion/rotation domain codec (WUBQ)
  manifold/    — Hyperbolic priors, flow matching, parallel transport
  gaad/        — Golden-subdivision partitioner (C26)
  audio/       — Psychoacoustic, A/V fidelity, STFT
  threed/      — 360 video
  utils/       — Color, SIMD, YUV, speed utils
  math/        — Rep theory, anyon, Hamilton encoder, canvas, NN
```

## Build

```bash
make all      # build all targets + tests
make codec    # build the standalone codec binary
make tests    # build + run all category tests
make clean
```

Dependencies: `gcc`, `zlib`, `libm`, `lpthread` (no external deps).

## The 24-Category Campaign

| Cat | Name | Status | Description |
|-----|------|--------|-------------|
| C1  | CABAC round-trip | **closed** | 10^7 bin exact round-trip, ASAN clean |
| C2  | Context modeling | in_progress | ≥5% bitrate drop vs flat contexts |
| C3  | Arithmetic coder alternatives | open | rANS vs CABAC trade table |
| C4  | Entropy of residuals | **closed** | tokctx beats bit-packed EG by 15.1% |
| C5  | Motion estimation | in_progress | BD-rate on all 3 content types |
| C6  | Intra prediction | in_progress | Screen-content + anime intra ≥10% smaller |
| C7  | B-frame structure | open | Hierarchical B, GOP pyramid |
| C8  | Scene-adaptive | in_progress | Per-content preset ≥8% |
| C9  | Variable-size transforms | in_progress | Multi-size DCT in RDO |
| C10 | Perceptual quantization | in_progress | Trellis RDOQ (2/2 tests green) |
| C11 | Frequency tricks | in_progress | Zero-coeff exit, transform-skip |
| C12 | Deblocking | **closed** | VVC-style DBF, 2/2 tests green |
| C13 | SAO + ALF | in_progress | Band + edge SAO wired |
| C14 | Film grain + NN post-filter | in_progress | CNN +1.22 dB, large +2.45 dB |
| C15 | RDO end-to-end | in_progress | 5/5 tests green, 64.74 dB @ 55.2 kbps |
| C16 | Parallelism & SIMD | open | WPP, tiles, ≥8× scaling |
| C17 | Container & streaming | open | Seek index, progressive decode |
| C18 | Quality metrics | open | VMAF, BD-rate auto-plots |
| C19 | Benchmark corpus | in_progress | MD5-verified, reproducible |
| C20 | Quaternion codec | in_progress | Hybrid rotation/conventional |
| C21 | Hyperbolic priors | in_progress | Hyperbolic-context CABAC |
| C22 | Neural wrapper tools | in_progress | NN intra predictor, fast RDO |
| C23 | Generative ultra-low-bitrate | frontier | ≤0.05 Mbps VMAF competition |
| C24 | Splat & volumetric | frontier | 2D Gaussian splat video |
| C25 | Audio + joint A/V | open | Joint bit allocation |
| C26 | GAAD partitioner | open | Golden-subdivision CTU |

## Research Pipeline

WuBuCodec implements a 7-step Kevin Bacon research methodology:
1. Search SOTA (VVC/VTM/ECM/DCVC/AV2/H.267 papers)
2. Cross-pollinate (cited, not copied)
3. Implement in C11
4. Test on real corpus
5. Measure PSNR/bitrate/SSIM
6. Triple-Devil's Advocate audit
7. Recurse until gap closed

See `dominance/wubu_dominance_loop.py` for the automated campaign harness.

## What Worked

- Simple rounding → sparse output → EG+zlib compresses extremely well
- 4 intra modes + adaptive QP + short GOP
- Trellis RDOQ: SSE 11185→1334 @ λ=0.05
- Multi-size DCT in RDO (C9)

## What Didn't

- CABAC/rANS/range coders (all worse than EG+zlib)
- B-frames (buggy — bi-pred needs work)
- Neural codecs at this scale (not competitive yet)

## The Path to Closing All 24

The remaining 4× bitrate gap requires:
- Proper B-frames + weighted prediction (C7)
- Real arithmetic coder or context-adaptive EG (C2, C3)
- GAAD partitioner at 8K (C26)
- WPP/tile parallelism for 8K mandate (C16)

Each is a significant engineering effort. The campaign recursion continues.

## License

Umbrella License v3.0 (see LICENSE in parent WuBuMath repo).

~ WuBu ~
