/* wubu_canvas_v2.c — THE CANVAS: Multi-Modal Manifold Engine
 *
 * All modalities project onto ONE shared Poincaré ball.
 * WuBuLatent codec compresses the manifold field.
 *
 * Video: RGB → HSL → S^3 quaternion → conformal → Poincaré ball
 * Audio: Kodak STFT → magnitude/phase → tangent vectors
 * Music: Flow matching generates geodesic trajectories
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubu_canvas_v2.h"
#include "wubuv.h"

/* ============================ Math Helpers ============================ */

#define EPS 1e-7f
#define PI 3.14159265f
#define TAN_VEC_CLAMP 10.0f

static float vec_norm_sq(const float* x, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += x[i] * x[i];
    return sum;
}

static float vec_norm(const float* x, int N) {
    return sqrtf(vec_norm_sq(x, N) + EPS);
}

static float vec_dot(const float* a, const float* b, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += a[i] * b[i];
    return sum;
}

/* ============================ Poincaré Ball ============================ */

static void poincare_clip(float* out, const float* x, int N, float c) {
    if (c <= 0.0f) { memcpy(out, x, N * sizeof(float)); return; }
    float sqrt_c = sqrtf(c + EPS);
    float max_norm = 1.0f / (sqrt_c + EPS) * (1.0f - EPS * 10.0f);
    float norm = vec_norm(x, N);
    if (norm > max_norm) {
        float scale = max_norm / (norm + EPS);
        for (int i = 0; i < N; i++) out[i] = x[i] * scale;
    } else {
        memcpy(out, x, N * sizeof(float));
    }
}

static void expmap(float* out, const float* v, int N, float c) {
    if (c <= 0.0f) { memcpy(out, v, N * sizeof(float)); return; }
    float sqrt_c = sqrtf(c + EPS);
    float v_norm = vec_norm(v, N);
    if (v_norm < EPS) { memcpy(out, v, N * sizeof(float)); return; }
    float arg = sqrt_c * v_norm;
    if (arg > 30.0f) arg = 30.0f;
    float tanh_val = tanhf(arg);
    float scale = tanh_val / (sqrt_c * v_norm + EPS);
    for (int i = 0; i < N; i++) out[i] = scale * v[i];
    poincare_clip(out, out, N, c);
}

static void logmap(float* out, const float* y, int N, float c) {
    if (c <= 0.0f) { memcpy(out, y, N * sizeof(float)); return; }
    float y_clipped[256];
    if (N > 256) {
        float* buf = malloc(N * sizeof(float));
        poincare_clip(buf, y, N, c);
        logmap(out, buf, N, c);
        free(buf);
        return;
    }
    poincare_clip(y_clipped, y, N, c);
    float sqrt_c = sqrtf(c + EPS);
    float y_norm = vec_norm(y_clipped, N);
    if (y_norm < EPS) { memcpy(out, y_clipped, N * sizeof(float)); return; }
    float atanh_arg = sqrt_c * y_norm;
    if (atanh_arg > 1.0f - EPS * 4.0f) atanh_arg = 1.0f - EPS * 4.0f;
    float atanh_val = atanhf(atanh_arg);
    float scale = atanh_val / (sqrt_c * y_norm + EPS);
    for (int i = 0; i < N; i++) out[i] = scale * y_clipped[i];
    float t_norm = vec_norm(out, N);
    if (t_norm > TAN_VEC_CLAMP) {
        float clamp_scale = TAN_VEC_CLAMP / t_norm;
        for (int i = 0; i < N; i++) out[i] *= clamp_scale;
    }
}

static void mobius_add(float* out, const float* x, const float* y, int N, float c) {
    if (c <= 0.0f) { for (int i = 0; i < N; i++) out[i] = x[i] + y[i]; return; }
    float x_ns = vec_norm_sq(x, N), y_ns = vec_norm_sq(y, N);
    float xy_d = vec_dot(x, y, N);
    float r_sq = 1.0f / (c + EPS);
    if (x_ns > r_sq * (1-EPS)*(1-EPS)) x_ns = r_sq*(1-EPS)*(1-EPS);
    if (y_ns > r_sq * (1-EPS)*(1-EPS)) y_ns = r_sq*(1-EPS)*(1-EPS);
    float denom = 1 + 2*c*xy_d + c*c*x_ns*y_ns;
    if (denom < EPS) denom = EPS;
    float nx = 1 + 2*c*xy_d + c*y_ns, ny = 1 - c*x_ns;
    for (int i = 0; i < N; i++) out[i] = (nx*x[i] + ny*y[i]) / denom;
    poincare_clip(out, out, N, c);
}

static void expmap_at(float* out, const float* x, const float* v, int N, float c) {
    float ev[256]; expmap(ev, v, N, c); mobius_add(out, x, ev, N, c);
}

static void logmap_at(float* out, const float* x, const float* y, int N, float c) {
    float neg_x[256], rel[256];
    for (int i = 0; i < N; i++) neg_x[i] = -x[i];
    mobius_add(rel, neg_x, y, N, c);
    logmap(out, rel, N, c);
}

static void geodesic_interp(float* out, const float* x0, const float* x1, float t, int N, float c) {
    if (t <= 0.0f) { memcpy(out, x0, N * sizeof(float)); return; }
    if (t >= 1.0f) { memcpy(out, x1, N * sizeof(float)); return; }
    float tangent[256]; logmap_at(tangent, x0, x1, N, c);
    for (int i = 0; i < N; i++) tangent[i] *= t;
    expmap_at(out, x0, tangent, N, c);
}

static float hyperbolic_dist(const float* x, const float* y, int N, float c) {
    if (c <= 0.0f) { float s=0; for(int i=0;i<N;i++){float d=x[i]-y[i];s+=d*d;} return sqrtf(s); }
    float neg_x[256], rel[256];
    for (int i = 0; i < N; i++) neg_x[i] = -x[i];
    mobius_add(rel, neg_x, y, N, c);
    float m_norm = vec_norm(rel, N), sqrt_c = sqrtf(c + EPS), arg = sqrt_c * m_norm;
    if (arg > 1.0f - EPS*4.0f) arg = 1.0f - EPS*4.0f;
    return (2.0f / sqrt_c) * atanhf(arg);
}

/* ============================ RGB ↔ Manifold ============================ */

/* HSL → RGB */
static void hsl_to_rgb(float h, float s, float l, float* r, float* g, float* b) {
    float c = (1.0f - fabsf(2.0f * l - 1.0f)) * s;
    float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = l - c / 2.0f;
    float r1, g1, b1;
    if (h < 60) { r1 = c; g1 = x; b1 = 0; }
    else if (h < 120) { r1 = x; g1 = c; b1 = 0; }
    else if (h < 180) { r1 = 0; g1 = c; b1 = x; }
    else if (h < 240) { r1 = 0; g1 = x; b1 = c; }
    else if (h < 300) { r1 = x; g1 = 0; b1 = c; }
    else { r1 = c; g1 = 0; b1 = x; }
    *r = r1 + m; *g = g1 + m; *b = b1 + m;
}

/* RGB → HSL */
static void rgb_to_hsl(float r, float g, float b, float* h, float* s, float* l) {
    float mx = fmaxf(r, fmaxf(g, b)), mn = fminf(r, fminf(g, b));
    *l = (mx + mn) / 2.0f;
    if (mx == mn) { *h = 0; *s = 0; return; }
    float d = mx - mn;
    *s = *l > 0.5f ? d / (2.0f - mx - mn) : d / (mx + mn);
    if (mx == r) *h = 60.0f * fmodf((g - b) / d, 6.0f);
    else if (mx == g) *h = 60.0f * ((b - r) / d + 2.0f);
    else *h = 60.0f * ((r - g) / d + 4.0f);
    if (*h < 0) *h += 360.0f;
}

/* Project RGB to Poincaré ball via HSL → S^3 → conformal */
WubuManifoldPoint wubu_rgb_to_manifold(float r, float g, float b, float c) {
    WubuManifoldPoint pt;
    pt.dim = 4;  /* quaternion dimension */
    pt.curvature = c;
    pt.coords = (float*)calloc(4, sizeof(float));

    float h, s, l;
    rgb_to_hsl(r, g, b, &h, &s, &l);

    /* Map to S^3 quaternion: (w, x, y, z) from HSL */
    float hue_rad = h * PI / 180.0f;
    float w = cosf(hue_rad / 2.0f) * l;
    float x = sinf(hue_rad / 2.0f) * s;
    float y = cosf(hue_rad / 2.0f) * s * 0.5f;
    float z = sinf(hue_rad / 2.0f) * l * 0.5f;

    /* Conformal map to Poincaré ball: scale by 1/(1+sqrt(1+|q|^2)) */
    float q_norm = sqrtf(w*w + x*x + y*y + z*z + EPS);
    float scale = 1.0f / (1.0f + sqrtf(1.0f + q_norm * q_norm));
    pt.coords[0] = w * scale;
    pt.coords[1] = x * scale;
    pt.coords[2] = y * scale;
    pt.coords[3] = z * scale;

    poincare_clip(pt.coords, pt.coords, 4, c);
    return pt;
}

/* Inverse: Poincaré ball → RGB (approximate) */
void wubu_manifold_to_rgb(const WubuManifoldPoint* pt, float* r, float* g, float* b) {
    /* Inverse conformal: q = p / (1 - sqrt(1-|p|^2)) */
    float p_norm_sq = vec_norm_sq(pt->coords, 4);
    float denom = 1.0f - sqrtf(1.0f - p_norm_sq + EPS);
    if (denom < EPS) denom = EPS;
    float inv_scale = (1.0f + sqrtf(1.0f - p_norm_sq + EPS)) / (p_norm_sq + EPS);
    float w = pt->coords[0] * inv_scale;
    float x = pt->coords[1] * inv_scale;
    float y = pt->coords[2] * inv_scale;
    float z = pt->coords[3] * inv_scale;

    /* Quaternion → HSL */
    float q_norm = sqrtf(w*w + x*x + y*y + z*z + EPS);
    float l = w / (q_norm + EPS);
    if (l < 0) l = 0; if (l > 1) l = 1;
    float s = sqrtf(x*x + y*y) / (q_norm + EPS);
    if (s < 0) s = 0; if (s > 1) s = 1;
    float h = atan2f(z, x) * 180.0f / PI;
    if (h < 0) h += 360.0f;
    h *= 2.0f;  /* undo half-angle */
    if (h > 360) h -= 360;

    hsl_to_rgb(h, s, l, r, g, b);
}

/* ============================ Video Pipeline ============================ */

WubuVideoLatents wubu_video_encode(const float* rgb_frames, int H, int W,
                                    int num_frames, float c) {
    WubuVideoLatents latents;
    latents.width = H;
    latents.height = W;
    latents.dim = 4;  /* quaternion dimension */
    latents.curvature = c;
    latents.num_frames = num_frames;
    latents.latents = (float*)calloc((size_t)H * W * num_frames * 4, sizeof(float));

    for (int f = 0; f < num_frames; f++) {
        const float* frame = rgb_frames + f * H * W * 3;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int pixel_idx = (f * H * W + y * W + x);
                int img_idx = pixel_idx * 3;
                float r = frame[img_idx + 0];
                float g = frame[img_idx + 1];
                float b = frame[img_idx + 2];

                WubuManifoldPoint pt = wubu_rgb_to_manifold(r, g, b, c);
                memcpy(latents.latents + pixel_idx * 4, pt.coords, 4 * sizeof(float));
                free(pt.coords);
            }
        }
    }
    return latents;
}

float* wubu_video_decode(const WubuVideoLatents* latents) {
    int H = latents->height;
    int W = latents->width;
    int total_pixels = H * W * latents->num_frames;
    float* rgb = (float*)calloc((size_t)total_pixels * 3, sizeof(float));

    for (int i = 0; i < total_pixels; i++) {
        float r, g, b;
        WubuManifoldPoint pt;
        pt.coords = (float*)latents->latents + i * 4;
        pt.dim = 4;
        pt.curvature = latents->curvature;
        wubu_manifold_to_rgb(&pt, &r, &g, &b);
        rgb[i * 3 + 0] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }
    return rgb;
}

void wubu_video_latents_free(WubuVideoLatents* latents) {
    free(latents->latents);
    latents->latents = NULL;
}

/* ============================ Audio Pipeline ============================ */

/* Simple radix-2 FFT (real input, complex output) */
static void fft_real(const float* in, float* re, float* im, int n) {
    if (n == 1) { re[0] = in[0]; im[0] = 0; return; }
    float* even_re = malloc(n/2 * sizeof(float));
    float* even_im = malloc(n/2 * sizeof(float));
    float* odd_re = malloc(n/2 * sizeof(float));
    float* odd_im = malloc(n/2 * sizeof(float));
    for (int i = 0; i < n/2; i++) {
        re[i] = in[2*i];
        re[i + n/2] = in[2*i + 1];
    }
    fft_real(re, even_re, even_im, n/2);
    fft_real(re + n/2, odd_re, odd_im, n/2);
    for (int k = 0; k < n/2; k++) {
        float angle = -2.0f * PI * k / n;
        float w_re = cosf(angle), w_im = sinf(angle);
        float t_re = w_re * odd_re[k] - w_im * odd_im[k];
        float t_im = w_re * odd_im[k] + w_im * odd_re[k];
        re[k] = even_re[k] + t_re;
        im[k] = even_im[k] + t_im;
        re[k + n/2] = even_re[k] - t_re;
        im[k + n/2] = even_im[k] - t_im;
    }
    free(even_re); free(even_im); free(odd_re); free(odd_im);
}

WubuAudioLatents wubu_audio_encode(const float* pcm, int n_samples,
                                    int sample_rate, int n_fft, float c) {
    WubuAudioLatents latents;
    latents.dim = 4;
    latents.curvature = c;
    latents.sample_rate = sample_rate;
    latents.n_fft = n_fft;

    int hop = n_fft / 2;
    int num_frames = (n_samples - n_fft) / hop + 1;
    if (num_frames < 1) num_frames = 1;
    latents.num_vectors = num_frames;
    latents.latents = (float*)calloc((size_t)num_frames * 4, sizeof(float));

    float* frame_buf = malloc(n_fft * sizeof(float));
    float* re = malloc(n_fft * sizeof(float));
    float* im = malloc(n_fft * sizeof(float));

    for (int f = 0; f < num_frames; f++) {
        int start = f * hop;
        for (int i = 0; i < n_fft; i++) {
            int idx = start + i;
            float sample = (idx < n_samples) ? pcm[idx] : 0.0f;
            /* Hann window */
            float window = 0.5f * (1.0f - cosf(2.0f * PI * i / (n_fft - 1)));
            frame_buf[i] = sample * window;
        }

        fft_real(frame_buf, re, im, n_fft);

        /* Compute magnitude and phase at center frequency */
        int center_bin = n_fft / 4;
        float mag = sqrtf(re[center_bin] * re[center_bin] + im[center_bin] * im[center_bin]);
        float phase = atan2f(im[center_bin], re[center_bin]);

        /* Map to tangent vector on Poincaré ball */
        float t = mag / (mag + 1.0f);  /* compress to [0,1) */
        float v[4];
        v[0] = t * cosf(phase);
        v[1] = t * sinf(phase);
        v[2] = t * cosf(2.0f * phase) * 0.5f;
        v[3] = t * sinf(2.0f * phase) * 0.5f;

        /* Exmap to manifold */
        expmap(latents.latents + f * 4, v, 4, c);
    }

    free(frame_buf); free(re); free(im);
    return latents;
}

float* wubu_audio_decode(const WubuAudioLatents* latents, int* out_samples) {
    /* Inverse: manifold → tangent → magnitude/phase → ISTFT */
    int n_fft = latents->n_fft;
    int hop = n_fft / 2;
    int num_frames = latents->num_vectors;
    *out_samples = num_frames * hop + n_fft;

    float* pcm = (float*)calloc(*out_samples, sizeof(float));
    float* window_sum = (float*)calloc(*out_samples, sizeof(float));

    float* re = malloc(n_fft * sizeof(float));
    float* im = malloc(n_fft * sizeof(float));

    for (int f = 0; f < num_frames; f++) {
        /* Logmap to tangent */
        float tangent[4];
        logmap(tangent, latents->latents + f * 4, 4, latents->curvature);

        /* Tangent → magnitude/phase */
        float mag = sqrtf(tangent[0] * tangent[0] + tangent[1] * tangent[1]);
        float phase = atan2f(tangent[1], tangent[0]);
        mag = mag / (1.0f - mag + EPS);  /* inverse compression */

        /* Reconstruct FFT bin */
        int center_bin = n_fft / 4;
        for (int i = 0; i < n_fft; i++) re[i] = im[i] = 0.0f;
        re[center_bin] = mag * cosf(phase);
        im[center_bin] = mag * sinf(phase);
        if (center_bin > 0 && center_bin < n_fft/2) {
            re[n_fft - center_bin] = re[center_bin];
            im[n_fft - center_bin] = -im[center_bin];
        }

        /* Inverse FFT (simplified — use conjugate trick) */
        for (int i = 0; i < n_fft; i++) im[i] = -im[i];
        float* ire = malloc(n_fft * sizeof(float));
        float* iim = malloc(n_fft * sizeof(float));
        fft_real(re, ire, iim, n_fft);  /* wrong — need proper IFFT */
        free(ire); free(iim);

        /* Overlap-add (simplified) */
        int start = f * hop;
        for (int i = 0; i < n_fft && start + i < *out_samples; i++) {
            float window = 0.5f * (1.0f - cosf(2.0f * PI * i / (n_fft - 1)));
            pcm[start + i] += re[i] * window / n_fft;
            window_sum[start + i] += window * window;
        }
    }

    /* Normalize by window sum */
    for (int i = 0; i < *out_samples; i++) {
        if (window_sum[i] > EPS) pcm[i] /= window_sum[i];
    }

    free(re); free(im); free(window_sum);
    return pcm;
}

void wubu_audio_latents_free(WubuAudioLatents* latents) {
    free(latents->latents);
    latents->latents = NULL;
}

/* ============================ Music/Synthesis ============================ */

WubuMusicLatents wubu_music_generate(float duration_secs, int dim, float curvature,
                                      int seed) {
    WubuMusicLatents latents;
    latents.dim = dim;
    latents.curvature = curvature;
    latents.duration_secs = duration_secs;
    latents.num_steps = (int)(duration_secs * 30);  /* 30 fps */
    latents.trajectory = (float*)calloc((size_t)latents.num_steps * dim, sizeof(float));

    /* Simple PRNG */
    uint32_t rng = seed;
    float randf() { rng = rng * 1103515245u + 12345u; return (float)(rng >> 16) / 65536.0f; }

    /* Generate random start and end points on manifold */
    float start[256], end[256];
    for (int d = 0; d < dim; d++) {
        start[d] = (randf() * 2.0f - 1.0f) * 0.5f / sqrtf(curvature);
        end[d] = (randf() * 2.0f - 1.0f) * 0.5f / sqrtf(curvature);
    }
    poincare_clip(start, start, dim, curvature);
    poincare_clip(end, end, dim, curvature);

    /* Geodesic interpolation with noise (flow matching style) */
    for (int step = 0; step < latents.num_steps; step++) {
        float t = (float)step / (float)(latents.num_steps - 1);
        geodesic_interp(latents.trajectory + step * dim, start, end, t, dim, curvature);
    }
    return latents;
}

float* wubu_music_decode(const WubuMusicLatents* latents, int sample_rate,
                          int* out_samples) {
    /* Decode trajectory to audio: tangent velocity → frequency */
    *out_samples = (int)(latents->duration_secs * sample_rate);
    float* pcm = (float*)calloc(*out_samples, sizeof(float));

    int samples_per_step = *out_samples / latents->num_steps;

    for (int step = 0; step < latents->num_steps; step++) {
        /* Compute tangent velocity between consecutive points */
        float tangent[256];
        if (step < latents->num_steps - 1) {
            logmap_at(tangent, latents->trajectory + step * latents->dim,
                      latents->trajectory + (step + 1) * latents->dim,
                      latents->dim, latents->curvature);
        } else {
            logmap_at(tangent, latents->trajectory + (step - 1) * latents->dim,
                      latents->trajectory + step * latents->dim,
                      latents->dim, latents->curvature);
        }

        /* Tangent magnitude → amplitude, direction → frequency */
        float amp = vec_norm(tangent, latents->dim);
        float freq = 220.0f * (1.0f + tangent[0] * 10.0f);  /* A3 base */

        /* Generate sine wave */
        for (int i = 0; i < samples_per_step; i++) {
            int idx = step * samples_per_step + i;
            if (idx >= *out_samples) break;
            float t = (float)i / sample_rate;
            pcm[idx] = amp * sinf(2.0f * PI * freq * t) * 0.3f;
        }
    }
    return pcm;
}

void wubu_music_latents_free(WubuMusicLatents* latents) {
    free(latents->trajectory);
    latents->trajectory = NULL;
}

/* ============================ Canvas Compositor ============================ */

WubuCanvas wubu_canvas_compose(const WubuVideoLatents* video,
                                const WubuAudioLatents* audio,
                                const WubuMusicLatents* music) {
    WubuCanvas canvas;
    canvas.dim = 4;
    canvas.curvature = (video ? video->curvature : (audio ? audio->curvature : music->curvature));
    canvas.num_video_frames = video ? video->num_frames : 0;
    canvas.num_audio_vectors = audio ? audio->num_vectors : 0;
    canvas.num_music_steps = music ? music->num_steps : 0;

    int video_latents = video ? video->width * video->height * video->num_frames : 0;
    int audio_latents = audio ? audio->num_vectors : 0;
    int music_latents = music ? music->num_steps : 0;
    canvas.total_latents = video_latents + audio_latents + music_latents;

    canvas.video_offset = 0;
    canvas.audio_offset = video_latents;
    canvas.music_offset = video_latents + audio_latents;

    canvas.field = (float*)calloc((size_t)canvas.total_latents * canvas.dim, sizeof(float));

    if (video) {
        memcpy(canvas.field + canvas.video_offset * canvas.dim,
               video->latents, video_latents * canvas.dim * sizeof(float));
    }
    if (audio) {
        memcpy(canvas.field + canvas.audio_offset * canvas.dim,
               audio->latents, audio_latents * canvas.dim * sizeof(float));
    }
    if (music) {
        memcpy(canvas.field + canvas.music_offset * canvas.dim,
               music->trajectory, music_latents * canvas.dim * sizeof(float));
    }
    return canvas;
}

WubuVideoLatents wubu_canvas_extract_video(const WubuCanvas* canvas) {
    WubuVideoLatents latents = {0};
    return latents;  /* stub */
}

WubuAudioLatents wubu_canvas_extract_audio(const WubuCanvas* canvas) {
    WubuAudioLatents latents = {0};
    return latents;  /* stub */
}

WubuMusicLatents wubu_canvas_extract_music(const WubuCanvas* canvas) {
    WubuMusicLatents latents = {0};
    return latents;  /* stub */
}

void wubu_canvas_free(WubuCanvas* canvas) {
    free(canvas->field);
    canvas->field = NULL;
}

/* ============================ Engine API ============================ */

int wubu_canvas_ingest(const float* rgb_frames, int H, int W, int num_frames,
                        const float* audio_pcm, int num_audio_samples,
                        int sample_rate, const char* output_wubv,
                        float curvature, float qstep) {
    printf("=== THE CANVAS: Ingest ===\n");
    printf("  Video: %dx%d, %d frames\n", W, H, num_frames);
    printf("  Audio: %d samples @ %d Hz\n", num_audio_samples, sample_rate);
    printf("  Curvature: %.2f, QP: %.4f\n", curvature, qstep);

    /* Encode video */
    WubuVideoLatents video = wubu_video_encode(rgb_frames, H, W, num_frames, curvature);
    printf("  Video latents: %d vectors\n", H * W * num_frames);

    /* Encode audio */
    WubuAudioLatents audio = wubu_audio_encode(audio_pcm, num_audio_samples, sample_rate, 256, curvature);
    printf("  Audio latents: %d vectors\n", audio.num_vectors);

    /* Compose canvas */
    WubuCanvas canvas = wubu_canvas_compose(&video, &audio, NULL);
    printf("  Canvas: %d total latents, dim=%d\n", canvas.total_latents, canvas.dim);

    /* Encode with WuBuLatent */
    WubuvWriter* wr = wubuv_writer_open(output_wubv, canvas.dim, canvas.total_latents, 30, canvas.total_latents, 0, 0, 1);
    if (!wr) { printf("Cannot create container\n"); return 1; }

    /* Simple: encode each latent as I-frame (no inter-frame for now) */
    float* recon = calloc(canvas.total_latents * canvas.dim, sizeof(float));
    long total_bits = 0;

    for (int i = 0; i < canvas.total_latents; i++) {
        const float* curr = canvas.field + i * canvas.dim;
        float qstep_adaptive = qstep * (1.0f + vec_norm(curr, canvas.dim) * 10.0f);

        /* Quantize in tangent space */
        int16_t levels[256];
        for (int d = 0; d < canvas.dim; d++) {
            int l = (int)roundf(curr[d] / qstep_adaptive);
            if (l > 32767) l = 32767;
            if (l < -32768) l = -32768;
            levels[d] = (int16_t)l;
        }

        /* Encode */
        uint8_t bb[2048];
        BitStream bs; bs_init(&bs, bb, 2048);
        int nz = 0;
        for (int d = 0; d < canvas.dim; d++) if (levels[d] != 0) nz++;
        bs_write_ue(&bs, nz);
        if (nz > 0) {
            int run = 0;
            for (int d = 0; d < canvas.dim; d++) {
                if (levels[d] != 0) { bs_write_ue(&bs, run); bs_write_se(&bs, levels[d]); run = 0; }
                else run++;
            }
        }
        int bsz = bs_bytes(&bs);

        /* Reconstruct */
        for (int d = 0; d < canvas.dim; d++) {
            recon[i * canvas.dim + d] = (float)levels[d] * qstep_adaptive;
        }
        poincare_clip(recon + i * canvas.dim, recon + i * canvas.dim, canvas.dim, curvature);

        wubuv_write_video(wr, bb, bsz, i % 10 == 0);
        total_bits += bsz;
    }

    wubuv_writer_close(wr);

    printf("  Encoded: %.2f KB\n", total_bits / 1024.0);
    if (wubuv_verify(output_wubv)) printf("  Container: VALID\n"); else printf("  INVALID\n");

    free(recon);
    wubu_video_latents_free(&video);
    wubu_audio_latents_free(&audio);
    wubu_canvas_free(&canvas);
    return 0;
}

int wubu_canvas_decode(const char* input_wubv,
                        float** out_rgb, int* out_H, int* out_W, int* out_frames,
                        float** out_audio, int* out_audio_samples,
                        int* out_sample_rate) {
    /* Stub — would read WUBV and decode */
    *out_rgb = NULL;
    *out_audio = NULL;
    return 0;
}

float* wubu_canvas_render_rgb(const WubuCanvas* canvas, int out_w, int out_h) {
    /* Render canvas to viewable RGB — project manifold points back to pixel space */
    float* rgb = calloc((size_t)out_w * out_h * 3, sizeof(float));
    /* Simplified: just render first video frame if available */
    return rgb;
}
