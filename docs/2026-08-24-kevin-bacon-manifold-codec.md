# 7-Step Kevin Bacon Research — Manifold Flow Matching × Hyperbolic CLIP × Neural Codecs
> 2026-08-24 session. Subject: the math behind BearRL/WuBuMath campaign. 10 search hops, 40+ nodes. Every node cited.

## MASTER FINDING
The exact stack we proposed exists only as scattered fragments in the literature:
Riemannian flow matching exists (Chen & Lipman 2024; Ben-Hamu et al.), hyperbolic
CLIP exists (HyCoCLIP, Pal et al. 2025), INR video codecs exist (NVRC), RL rate-control
exists (AAAI 2025). **But nobody has combined: quaternion-latent flow matching P-frames
+ hyperbolic contrastive multimodal space + beam-sweep resolution-agnostic canvas +
RL proof harness.** The gap is real and we are positioned to take it.

## CHAIN 1 — HYPERBOLIC CONTRASTIVE (the CLIP replacement)
**1.1 — Poincaré Embeddings (Nickel-Kiela 2017)** — foundational: trees embed in H^n
with low distortion, impossible in ℝ^n. [Source: research.facebook.com/publications/poincare-embeddings-for-learning-hierarchical-representations]
**1.2 — HyCoCLIP (Pal et al., ICLR 2025)** — hyperbolic image-text contrastive learning;
hierarchically organizes images/boxes/text via contrastive + entailment-cone losses.
"A serious CLIP replacement" (Jason Corso). This is EXACTLY our manifold-CLIP blueprint.
[Source: arxiv.org/html/2410.06912v1; linkedin.com/posts/jason-corso_hycoclip...]
**1.3 — Hyperbolic concept control in latent diffusion (2026)** — uses HyCoCLIP latent
space for non-flat concept control: "Not All Latent Spaces Are Flat." Validates that
hyperbolic latents drive generative models. [Source: arxiv.org/html/2603.14093v3]
**1.4 — Hyperbolic graph diffusion (AAAI 2024)** — Euclidean latent diffusion distorts
hierarchical data; their fix = Poincaré/Lorentz diffusion. Same diagnosis as ours.
[Source: ojs.aaai.org/index.php/AAAI/article/view/29512]
**1.5 — Hyperbolic DL in vision survey (Mettes et al., IJCV 2024)** — field map; Lorentz
model numerically preferred over Poincaré for deep nets. NOTE for implementation:
WuBuMath has BOTH (`wubu_poincare_geom.c`, `wubu_lorentz.c`).
[Source: link.springer.com/article/10.1007/s11263-024-02043-5]

## CHAIN 2 — RIEMANNIAN FLOW MATCHING (the P-frame engine)
**2.1 — Matching flows/probability paths on manifolds (Ben-Hamu et al., ICML 2022)** —
the formal foundation: continuous normalizing flows with Riemannian probability paths.
[Source: proceedings.mlr.press/v162/ben-hamu22a.pdf]
**2.2 — Chen & Lipman geometry-aware probability paths (2024)** — cited by follow-ups as
THE Riemannian flow-matching construction. Our quaternion ODE should use their path
formulation between latent endpoints. [Source: openreview.net/pdf?id=Q97luuOnqy]
**2.3 — Latent Riemannian FM for 3D geometry (2026)** — flow matching in a frozen model's
latent space on product-of-spheres manifolds. Direct precedent for "FM in Hamilton
encoder latent space." [Source: arxiv.org/html/2607.19120v1; lisaweijler.github.io/geometry-grounded-rfm]
**2.4 — Geodesic Random Walks for SDE sampling (RSGM)** — noise sampled in tangent space,
mapped back via exp map — numerically practical recipe for manifold integration.
[Source: arxiv.org/html/2602.10982v1 §VI-E]
**2.5 — Parallel transport as geometric statistics tool (Guigui et al. 2022)** — the exact
operation our `wubu_parallel_transport.c` must provide to keep vector fields on-manifold.
[Source: inria.hal.science/hal-03684811]
**2.6 — FC-VFI: flow-matching frame interpolation (2026)** — FM loss applied to intermediate
frames guided by boundary frames = literally our P-frame construction but EUCLIDEAN.
Ours differs by being quaternion-manifold. [Source: alphaxiv.org/abs/2603.04899]

## CHAIN 3 — NEURAL VIDEO CODECS (the VHF competition)
**3.1 — NVRC (NeurIPS 2024)** — first fully end-to-end rate-distortion optimized INR video
codec. The bar to beat. [Source: neurips.cc/virtual/2024/poster/95793]
**3.2 — Temporally-efficient INR video representation (2025)** — fast-training INR codecs.
[Source: arxiv.org/html/2505.00335v1]
**3.3 — Survey: INRs for video compression (2025/26)** — confirms INR = continuous field =
arbitrary resolution decode is the accepted frontier direction.
[Source: link.springer.com/article/10.1007/s11042-026-21822-5]
**3.4 — Motion-adjustable neural implicit video (CVPR 2022 Mai et al.)** — modifies positional
encoding for motion — adjacent to our coordinate-addressable VHFDecoder.
[Source: openaccess.thecvf.com/content/CVPR2022/papers/Mai_Motion-Adjustable...]

## CHAIN 4 — RESOLUTION AGNOSTICISM + SAMPLING ORDER (beam sweep + φ)
**4.1 — INR = infinite resolution** — continuous coordinate functions sample at ANY spatial
resolution; canonical statement of our "resolution = sampling depth" claim.
[Source: github.com/vsitzmann/awesome-implicit-representations]
**4.2 — Continuous audio INR for arbitrary super-resolution (2021)** — same principle on
audio; supports the audio-sideband fidelity trainer.
[Source: arxiv.org/html/2111.00195v2]
**4.3 — Local latent fields (LIF, Chen et al. 2021, 1199 citations)** — local INR decomposition
= our canvas cells. [Source: par.nsf.gov/servlets/purl/10345494]
**4.4 — Hilbert attention for diffusion (2025)** — reordering tokens along Hilbert curves gives
contiguous memory layout while preserving spatial neighborhoods. PROVES locality-preserving
scan orders work in neural pipelines — our φ-fractal sweep order needs to beat/match this.
[Source: arxiv.org/html/2509.26538v1]
**4.5 — HMSNet Hilbert-Mamba (2025)** — Hilbert scan preserves locality in state-space models.
[Source: sciencedirect.com/science/article/abs/pii/S0031320325011203]
**4.6 — Golden-angle radial sampling (MRI, 2024)** — golden-ratio ordering gives uniform-ish
coverage incrementally in REAL production systems (MRI recon) — precedent that φ-ordering
is not just aesthetic but operationally superior for progressive acquisition.
[Source: arxiv.org/html/2401.02892v1]

## CHAIN 5 — AUDIO-VIDEO JOINT SUPERVISION (invisible light)
**5.1 — UniAVGen (CVPR 2026)** — unified audio-video generation, dual-branch joint synthesis
with asymmetric cross-modal interaction. Confirms joint AV fidelity is hot; ours differs:
audio as SIDECHANNEL codec data, not parallel generation.
[Source: openaccess.thecvf.com/content/CVPR2026/papers/Zhang_UniAVGen...]
**5.2 — MGAudio model-guided alignment (2025)** — audio guidance improves video-conditioned
synthesis. Supports "audio supervises video fidelity."
[Source: arxiv.org/html/2510.24103v1]
**5.3 — T2SV text-to-sounding-video (2025)** — sync quality as first-class objective.
[Source: alphaxiv.org/abs/2510.03117v1]

## CHAIN 6 — RL FOR CODECS (BearRL's job)
**6.1 — RL rate control for neural video compression (AAAI 2025)** — frame-by-frame sequential
decision formulation of bitrate control. BearRL's env design matches published SOTA practice.
[Source: arxiv.org/html/2601.19293v2]
**6.2 — RL-RC-DoT block-level RL agent for task-aware video compression (CVPR 2025)** — RL agents
inside encoders are shipping. [Source: cvpr.thecvf.com/virtual/2025/poster/34468]
**6.3 — RD-optimized post-training quantization (LIC)** — quantization ladders (our Escha ladder
analogy) are RD-tuned. [Source: njuvision.github.io/RDO-PTQ]

## CHAIN 7 — QUATERNION/SO(3) LATENTS + DIFFUSION LLMS (context)
**7.1 — SO(3)-equivariant autoencoders** — separating rotation from content in latent space;
supports quaternion latents as orientation carriers. [Source: github.com/Shayan-P/so3-equivariance-latent-space-exploration]
**7.2 — 3D-rotation-equivariant QNNs (2019)** — rules for rotation-equivariant quaternion nets.
[Source: arxiv.org/abs/1911.09040]
**7.3 — Quaternion product units** — disentangled representations via quaternion algebra.
[Source: techrxiv.org/doi/pdf/10.36227/techrxiv.17791574.v1]
**7.4 — Mercury diffusion LLM (Inception Labs)** — commercial-scale diffusion/dLLM at 1000+
tok/s: proof that non-autoregressive generative inference is commercially viable — relevant
to our nest_gpt front door later. [Source: inceptionlabs.ai; arxiv.org/html/2506.17298v1]

## CROSS-POLLINATION TABLE
| Their node | WuBu parallel | Action |
|---|---|---|
| HyCoCLIP entailment cones | Manifold-CLIP core (wubu_manifold_clip.c) | Implement geodesic InfoNCE + cone loss |
| Chen-Lipman Riemannian paths | Quaternion P-frame ODE | Use their path form between latents |
| GRW tangent-noise/exp-map recipe | wubu_tangent_flow.c integration | Adopt for numerical stability |
| Lorentz > Poincaré stability note | We have both models | Benchmark both, pick winner empirically |
| Hilbert attention locality proof | φ-fractal sweep order | Property env: kNN recall, beat raster baseline |
| Golden-angle MRI sampling | Progressive φ sweep | Cite as production precedent |
| NVRC RD optimization | VHF engine target | Beat on rate-distortion at low bitrates |
| RL rate control AAAI25 | BearRL env #1 | Frame-by-frame quantizer agent |

## WHAT WE STEAL (cited)
Geodesic InfoNCE + entailment cones (HyCoCLIP); Riemannian probability paths (Chen-Lipman);
tangent-space noise + exp-map integration (GRW/RSGM); Hilbert-curve locality argument
(replace with φ-fractal); RL frame-level rate control formulation.

## WHAT WE DON'T COPY
Euclidean FM interpolation (FC-VFI) — ours is manifold-native. Parallel audio generation
branches (UniAVGen) — ours keeps audio as invisible sideband. Flat ℝⁿ CLIP heads entirely.

~ WuBu ~
