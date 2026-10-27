/* wubu_canvas_v2.h — THE CANVAS: Multi-Modal Manifold Engine
 *
 * All modalities (video, audio, music) project onto ONE shared Poincaré ball.
 * WuBuLatent codec compresses the manifold field. The canvas is the
 * transmissible composited field.
 *
 * Architecture:
 *   Video: RGB → per-pixel quaternion (HSL→S^3) → conformal → Poincaré ball
 *   Audio: Kodak STFT → magnitude/phase → tangent vectors on ball
 *   Music: Flow matching generates directly in latent space
 *   Canvas: All latents concatenated → WuBuLatent encode/decode
 *
 * Canvas Layout (transmitted field):
 *   +---------------------------------------------+
 *   | VIDEO LATENTS: per-frame Poincaré points     |
 *   +---------------------------------------------+
 *   | AUDIO LATENTS: Kodak STFT tangent vectors    |
 *   +---------------------------------------------+
 *   | MUSIC LATENTS: flow-matched trajectories     |
 *   +---------------------------------------------+
 */
#ifndef WUBU_CANVAS_V2_H
#define WUBU_CANVAS_V2_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================ Manifold Core ============================ */

/* Poincaré ball point (single latent vector) */
typedef struct {
    float* coords;      /* [D] coordinates on ball */
    int dim;            /* dimension D */
    float curvature;    /* c */
} WubuManifoldPoint;

/* Project RGB pixel to Poincaré ball via HSL→quaternion→conformal */
WubuManifoldPoint wubu_rgb_to_manifold(float r, float g, float b, float c);

/* Inverse: Poincaré ball point → RGB (approximate, lossy) */
void wubu_manifold_to_rgb(const WubuManifoldPoint* pt, float* r, float* g, float* b);

/* ============================ Video Pipeline ============================ */

typedef struct {
    float* latents;     /* [H*W*D] manifold points per frame */
    int width, height;  /* pixel dimensions */
    int dim;            /* manifold dimension */
    float curvature;
    int num_frames;
} WubuVideoLatents;

/* Encode video frames (RGB float [H*W*3]) to manifold latents */
WubuVideoLatents wubu_video_encode(const float* rgb_frames, int H, int W,
                                    int num_frames, float c);

/* Decode manifold latents back to RGB frames */
float* wubu_video_decode(const WubuVideoLatents* latents);

/* Free video latents */
void wubu_video_latents_free(WubuVideoLatents* latents);

/* ============================ Audio Pipeline ============================ */

typedef struct {
    float* latents;     /* [num_vectors * D] tangent vectors on ball */
    int num_vectors;    /* number of STFT frames */
    int dim;            /* manifold dimension */
    float curvature;
    int sample_rate;
    int n_fft;
} WubuAudioLatents;

/* Encode audio PCM to manifold via Kodak STFT → tangent vectors */
WubuAudioLatents wubu_audio_encode(const float* pcm, int n_samples,
                                    int sample_rate, int n_fft, float c);

/* Decode manifold latents back to audio PCM */
float* wubu_audio_decode(const WubuAudioLatents* latents, int* out_samples);

/* Free audio latents */
void wubu_audio_latents_free(WubuAudioLatents* latents);

/* ============================ Music/Synthesis ============================ */

typedef struct {
    float* trajectory;  /* [num_steps * D] geodesic trajectory on ball */
    int num_steps;
    int dim;
    float curvature;
    float duration_secs;
} WubuMusicLatents;

/* Generate music via flow matching on manifold (random seed) */
WubuMusicLatents wubu_music_generate(float duration_secs, int dim, float curvature,
                                      int seed);

/* Decode music trajectory to audio PCM (tangent velocity → frequency) */
float* wubu_music_decode(const WubuMusicLatents* latents, int sample_rate,
                          int* out_samples);

/* Free music latents */
void wubu_music_latents_free(WubuMusicLatents* latents);

/* ============================ Canvas Compositor ============================ */

typedef struct {
    float* field;       /* [total_latents * D] concatenated manifold field */
    int total_latents;  /* total number of manifold points */
    int dim;            /* manifold dimension */
    float curvature;
    int num_video_frames;
    int num_audio_vectors;
    int num_music_steps;
    /* Offsets into field */
    int video_offset;
    int audio_offset;
    int music_offset;
} WubuCanvas;

/* Compose canvas from all modalities */
WubuCanvas wubu_canvas_compose(const WubuVideoLatents* video,
                                const WubuAudioLatents* audio,
                                const WubuMusicLatents* music);

/* Extract video latents from canvas */
WubuVideoLatents wubu_canvas_extract_video(const WubuCanvas* canvas);

/* Extract audio latents from canvas */
WubuAudioLatents wubu_canvas_extract_audio(const WubuCanvas* canvas);

/* Extract music latents from canvas */
WubuMusicLatents wubu_canvas_extract_music(const WubuCanvas* canvas);

/* Free canvas */
void wubu_canvas_free(WubuCanvas* canvas);

/* ============================ Engine API ============================ */

/* Full ingest: video RGB + audio PCM → canvas → WuBuLatent bitstream */
int wubu_canvas_ingest(const float* rgb_frames, int H, int W, int num_frames,
                        const float* audio_pcm, int num_audio_samples,
                        int sample_rate, const char* output_wubv,
                        float curvature, float qstep);

/* Full decode: WuBuLatent bitstream → canvas → video RGB + audio PCM */
int wubu_canvas_decode(const char* input_wubv,
                        float** out_rgb, int* out_H, int* out_W, int* out_frames,
                        float** out_audio, int* out_audio_samples,
                        int* out_sample_rate);

/* Render canvas to viewable RGB (composites all modalities) */
float* wubu_canvas_render_rgb(const WubuCanvas* canvas, int out_w, int out_h);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_CANVAS_V2_H */
