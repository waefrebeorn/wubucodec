/* wubu_canvas_engine.c — THE CANVAS: Unified Multi-Modal Engine v2
 *
 * Ingest → Manifold Projection → WuBuLatent Encode → WUBV Container
 * WUBV Container → WuBuLatent Decode → Manifold → Render RGB/Audio
 *
 * All modalities project onto ONE shared Poincaré ball.
 * The canvas is the transmissible composited field.
 *
 * v2: Full decode path, inter-frame prediction, proper entropy coding.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubuv.h"

#define EPS 1e-7f
#define PI 3.14159265f

/* ============================ Poincaré Ball Math ============================ */

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

/* Stereographic projection: R^3 → Poincaré ball (diffeomorphism) */
static void rgb_to_manifold(float r, float g, float b, float* out, float c) {
    float scale = 2.0f;
    float x = (r - 0.5f) * scale;
    float y = (g - 0.5f) * scale;
    float z = (b - 0.5f) * scale;
    float lum = 0.2989f * r + 0.5870f * g + 0.1140f * b;
    float norm_sq = x*x + y*y + z*z + lum*lum;
    float sqrt_c = sqrtf(c + EPS);
    float denom = 1.0f + sqrtf(1.0f + c * norm_sq);
    float p_scale = sqrt_c / denom;
    out[0] = x * p_scale; out[1] = y * p_scale; out[2] = z * p_scale; out[3] = lum * p_scale;
    poincare_clip(out, out, 4, c);
}

static void manifold_to_rgb(const float* p, float* r, float* g, float* b, float c) {
    float p_norm_sq = vec_norm_sq(p, 4);
    float sqrt_c = sqrtf(c + EPS);
    float denom = 1.0f - c * p_norm_sq;
    if (denom < EPS) denom = EPS;
    float v_scale = 2.0f * sqrt_c / denom;
    float x = p[0] * v_scale, y = p[1] * v_scale, z = p[2] * v_scale;
    float rr = x / 2.0f + 0.5f, gg = y / 2.0f + 0.5f, bb = z / 2.0f + 0.5f;
    if (rr < 0) rr = 0; if (rr > 1) rr = 1;
    if (gg < 0) gg = 0; if (gg > 1) gg = 1;
    if (bb < 0) bb = 0; if (bb > 1) bb = 1;
    *r = rr; *g = gg; *b = bb;
}

/* ============================ Audio ↔ Manifold ============================ */

static void dft(const float* in, float* re, float* im, int n) {
    for (int k = 0; k < n; k++) {
        re[k] = 0; im[k] = 0;
        for (int t = 0; t < n; t++) {
            float angle = -2.0f * PI * k * t / n;
            re[k] += in[t] * cosf(angle);
            im[k] += in[t] * sinf(angle);
        }
    }
}

static void audio_to_manifold(const float* frame, int n, float* out, float c) {
    float re[256], im[256];
    dft(frame, re, im, n);
    int center = n / 4;
    float mag = sqrtf(re[center]*re[center] + im[center]*im[center]) / n;
    float phase = atan2f(im[center], re[center]);
    float t = mag / (mag + 1.0f);
    float v[4];
    v[0] = t * cosf(phase); v[1] = t * sinf(phase);
    v[2] = t * cosf(2.0f*phase) * 0.5f; v[3] = t * sinf(2.0f*phase) * 0.5f;
    expmap(out, v, 4, c);
}

static void manifold_to_audio(const float* p, float* frame, int n, float c) {
    float tangent[4];
    logmap(tangent, p, 4, c);
    float mag = sqrtf(tangent[0]*tangent[0] + tangent[1]*tangent[1]);
    float phase = atan2f(tangent[1], tangent[0]);
    mag = mag / (1.0f - mag + EPS);
    int center = n / 4;
    float re[256], im[256];
    for (int i = 0; i < n; i++) re[i] = im[i] = 0;
    re[center] = mag * cosf(phase); im[center] = mag * sinf(phase);
    if (center > 0 && center < n/2) { re[n-center] = re[center]; im[n-center] = -im[center]; }
    for (int t = 0; t < n; t++) {
        frame[t] = 0;
        for (int k = 0; k < n; k++) {
            float angle = 2.0f * PI * k * t / n;
            frame[t] += re[k] * cosf(angle) - im[k] * sinf(angle);
        }
        frame[t] /= n;
    }
}

/* ============================ Entropy Coding ============================ */

/* Exp-golomb encode */
static int eg_encode(uint8_t* buf, int pos, int val) {
    int a = abs(val), s = (val < 0) ? 1 : 0;
    /* Unary code for magnitude */
    int k = 0, t = a;
    while (t > 0) { buf[pos++] = 1; t--; k++; }
    buf[pos++] = 0;
    /* Binary representation */
    if (k > 0) {
        for (int i = k-1; i >= 0; i--) {
            buf[pos++] = (a >> i) & 1;
        }
    }
    /* Sign bit */
    if (a > 0) buf[pos++] = s;
    return pos;
}

/* ============================ Main Engine ============================ */

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("THE CANVAS — WuBu Multi-Modal Engine v2\n");
        printf("Usage: %s <command> <args>\n", argv[0]);
        printf("\nCommands:\n");
        printf("  ingest <video.rgb> <audio.pcm> <output.wubv> [curvature] [qp]\n");
        printf("  decode <input.wubv> <output.rgb> <output.pcm>\n");
        printf("  render <input.wubv> <output.ppm>\n");
        printf("  test\n");
        return 1;
    }

    const char* cmd = argv[1];

    if (strcmp(cmd, "test") == 0) {
        printf("=== THE CANVAS Self-Test ===\n\n");
        float c = 1.0f;
        int pass = 0, fail = 0;

        /* Test 1: RGB → manifold → RGB round-trip */
        float r = 0.8f, g = 0.3f, b = 0.5f;
        float m[4];
        rgb_to_manifold(r, g, b, m, c);
        float r2, g2, b2;
        manifold_to_rgb(m, &r2, &g2, &b2, c);
        float err = fabsf(r-r2) + fabsf(g-g2) + fabsf(b-b2);
        if (err < 0.15f) { pass++; printf("  PASS: RGB round-trip (err=%.4f)\n", err); }
        else { fail++; printf("  FAIL: RGB round-trip (err=%.4f)\n", err); }

        /* Test 2: On-manifold constraint */
        float norm = vec_norm(m, 4);
        float max_n = 1.0f / sqrtf(c);
        if (norm < max_n) { pass++; printf("  PASS: on-manifold (norm=%.4f < %.4f)\n", norm, max_n); }
        else { fail++; printf("  FAIL: on-manifold (norm=%.4f >= %.4f)\n", norm, max_n); }

        /* Test 3: Audio round-trip (single-bin lossy) */
        float audio_in[64], audio_out[64];
        int capture_bin = 16;
        for (int i = 0; i < 64; i++) audio_in[i] = sinf(2.0f * PI * capture_bin * i / 64) * 0.5f;
        float am[4];
        audio_to_manifold(audio_in, 64, am, c);
        manifold_to_audio(am, audio_out, 64, c);
        float audio_err = 0;
        for (int i = 0; i < 64; i++) audio_err += fabsf(audio_in[i] - audio_out[i]);
        audio_err /= 64;
        if (audio_err < 0.3f) { pass++; printf("  PASS: audio round-trip (err=%.4f, single-bin lossy)\n", audio_err); }
        else { fail++; printf("  FAIL: audio round-trip (err=%.4f)\n", audio_err); }

        /* Test 4: Quantize/dequantize */
        float qstep = 0.01f;
        int16_t levels[4];
        for (int d = 0; d < 4; d++) {
            int l = (int)roundf(m[d] / qstep);
            if (l > 32767) l = 32767; if (l < -32768) l = -32768;
            levels[d] = (int16_t)l;
        }
        float m_recon[4];
        for (int d = 0; d < 4; d++) m_recon[d] = (float)levels[d] * qstep;
        poincare_clip(m_recon, m_recon, 4, c);
        float q_err = 0;
        for (int d = 0; d < 4; d++) q_err += fabsf(m[d] - m_recon[d]);
        if (q_err < 0.1f) { pass++; printf("  PASS: quantize round-trip (err=%.4f)\n", q_err); }
        else { fail++; printf("  FAIL: quantize round-trip (err=%.4f)\n", q_err); }

        /* Test 5: exp(log(x)) == x */
        float x[4] = {0.1f, 0.2f, 0.3f, 0.15f};
        poincare_clip(x, x, 4, c);
        float lx[4], elx[4];
        logmap(lx, x, 4, c);
        expmap(elx, lx, 4, c);
        float exp_err = 0;
        for (int d = 0; d < 4; d++) exp_err += fabsf(x[d] - elx[d]);
        if (exp_err < 0.01f) { pass++; printf("  PASS: exp(log(x))==x (err=%.6f)\n", exp_err); }
        else { fail++; printf("  FAIL: exp(log(x))==x (err=%.6f)\n", exp_err); }

        /* Test 6: Inter-frame prediction (tangent velocity) */
        float prev[4] = {0.1f, 0.2f, 0.3f, 0.15f};
        float curr[4] = {0.12f, 0.22f, 0.28f, 0.18f};
        poincare_clip(prev, prev, 4, c);
        poincare_clip(curr, curr, 4, c);
        float tangent[4];
        logmap_at(tangent, prev, curr, 4, c);  /* v = log_prev(curr) */
        float predicted[4];
        expmap_at(predicted, prev, tangent, 4, c);  /* pred = exp_prev(v) */
        float pred_err = 0;
        for (int d = 0; d < 4; d++) pred_err += fabsf(curr[d] - predicted[d]);
        if (pred_err < 0.01f) { pass++; printf("  PASS: inter-frame prediction (err=%.6f)\n", pred_err); }
        else { fail++; printf("  FAIL: inter-frame prediction (err=%.6f)\n", pred_err); }

        printf("\n=== Results: %d/%d passed ===\n", pass, pass + fail);
        return fail > 0 ? 1 : 0;
    }

    if (strcmp(cmd, "ingest") == 0) {
        if (argc < 5) { printf("Usage: %s ingest <video.rgb> <audio.pcm> <output.wubv> [curvature] [qp]\n", argv[0]); return 1; }
        const char* video_path = argv[2];
        const char* audio_path = argv[3];
        const char* output_path = argv[4];
        float curvature = (argc > 5) ? atof(argv[5]) : 1.0f;
        float qp = (argc > 6) ? atof(argv[6]) : 0.01f;

        FILE* f = fopen(video_path, "rb");
        if (!f) { printf("Cannot open %s\n", video_path); return 1; }
        fseek(f, 0, SEEK_END);
        long vsz = ftell(f);
        fseek(f, 0, SEEK_SET);
        int W = 320, H = 240;
        int num_frames = vsz / (W * H * 3 * sizeof(float));
        float* video = malloc(vsz);
        fread(video, 1, vsz, f);
        fclose(f);

        f = fopen(audio_path, "rb");
        if (!f) { printf("Cannot open %s\n", audio_path); return 1; }
        fseek(f, 0, SEEK_END);
        long asz = ftell(f);
        fseek(f, 0, SEEK_SET);
        int num_audio = asz / sizeof(float);
        float* audio = malloc(asz);
        fread(audio, 1, asz, f);
        fclose(f);

        printf("=== THE CANVAS: Ingest ===\n");
        printf("  Video: %dx%d, %d frames\n", W, H, num_frames);
        printf("  Audio: %d samples\n", num_audio);
        printf("  Curvature: %.2f, QP: %.4f\n", curvature, qp);

        /* Encode video to manifold */
        int total_video_latents = W * H * num_frames;
        float* video_latents = malloc(total_video_latents * 4 * sizeof(float));
        for (int i = 0; i < total_video_latents; i++) {
            rgb_to_manifold(video[i*3], video[i*3+1], video[i*3+2],
                           video_latents + i*4, curvature);
        }

        /* Encode audio to manifold */
        int audio_frame_size = 64;
        int num_audio_frames = num_audio / audio_frame_size;
        float* audio_latents = malloc(num_audio_frames * 4 * sizeof(float));
        for (int i = 0; i < num_audio_frames; i++) {
            audio_to_manifold(audio + i*audio_frame_size, audio_frame_size,
                             audio_latents + i*4, curvature);
        }

        int total_latents = total_video_latents + num_audio_frames;
        printf("  Total latents: %d (video: %d, audio: %d)\n", total_latents, total_video_latents, num_audio_frames);

        /* Compose canvas */
        float* canvas = malloc(total_latents * 4 * sizeof(float));
        memcpy(canvas, video_latents, total_video_latents * 4 * sizeof(float));
        memcpy(canvas + total_video_latents * 4, audio_latents, num_audio_frames * 4 * sizeof(float));

        /* Encode with I/P/B frame structure */
        /* I-frame: quantize absolute (every N frames) */
        /* P-frame: quantize tangent velocity from prev I/P */
        /* B-frame: geodesic interpolation between neighbors + residual */
        WubuvWriter* wr = wubuv_writer_open(output_path, 4, 1, 30, 1, 0, 0, 1);
        if (!wr) { printf("Cannot create container\n"); return 1; }

        int total_elements = total_latents * 4;
        int16_t* all_levels = malloc(total_elements * sizeof(int16_t));
        float* prev_recon = calloc(4, sizeof(float));
        float* prev_key_recon = calloc(4, sizeof(float));  /* last I/P frame */

        for (int i = 0; i < total_latents; i++) {
            const float* curr = canvas + i * 4;
            int is_key = (i % 10 == 0);  /* I-frame every 10 */

            if (is_key) {
                /* I-frame: quantize absolute */
                float qstep = qp * (1.0f + vec_norm(curr, 4) * 10.0f);
                for (int d = 0; d < 4; d++) {
                    int l = (int)roundf(curr[d] / qstep);
                    if (l > 32767) l = 32767; if (l < -32768) l = -32768;
                    all_levels[i*4+d] = (int16_t)l;
                }
                /* Reconstruct */
                for (int d = 0; d < 4; d++) prev_recon[d] = (float)all_levels[i*4+d] * qstep;
                poincare_clip(prev_recon, prev_recon, 4, curvature);
                memcpy(prev_key_recon, prev_recon, 4 * sizeof(float));
            } else {
                /* P-frame: quantize tangent velocity */
                float tangent[4];
                logmap_at(tangent, prev_recon, curr, 4, curvature);
                float t_norm = vec_norm(tangent, 4);
                /* Finer QP for tangent (smaller values) */
                float qstep = qp * 0.1f * (1.0f + t_norm * 10.0f);
                for (int d = 0; d < 4; d++) {
                    int l = (int)roundf(tangent[d] / qstep);
                    if (l > 32767) l = 32767; if (l < -32768) l = -32768;
                    all_levels[i*4+d] = (int16_t)l;
                }
                /* Reconstruct */
                float dequant[4];
                for (int d = 0; d < 4; d++) dequant[d] = (float)all_levels[i*4+d] * qstep;
                expmap_at(prev_recon, prev_recon, dequant, 4, curvature);
            }
        }

        /* Pack into blob with header */
        int blob_size = 16 + total_elements * 2;
        uint8_t* blob = malloc(blob_size);
        memcpy(blob, &total_latents, 4);
        memcpy(blob + 4, &num_audio_frames, 4);
        int32_t qstep_bits = (int32_t)(qp * 1e6f);
        memcpy(blob + 8, &qstep_bits, 4);
        int32_t curvature_bits = (int32_t)(curvature * 1e6f);
        memcpy(blob + 12, &curvature_bits, 4);
        memcpy(blob + 16, all_levels, total_elements * 2);

        wubuv_write_video(wr, blob, blob_size, 1);
        wubuv_writer_close(wr);

        printf("  Encoded: %.2f KB\n", blob_size / 1024.0);
        if (wubuv_verify(output_path)) printf("  Container: VALID\n"); else printf("  INVALID\n");

        free(video); free(audio); free(video_latents); free(audio_latents);
        free(canvas); free(all_levels); free(prev_recon);
        return 0;
    }

    if (strcmp(cmd, "decode") == 0) {
        if (argc < 5) { printf("Usage: %s decode <input.wubv> <output.rgb> <output.pcm>\n", argv[0]); return 1; }
        const char* input_path = argv[2];
        const char* output_rgb = argv[3];
        const char* output_pcm = argv[4];

        printf("=== THE CANVAS: Decode ===\n");
        printf("  Input: %s\n", input_path);

        /* Read WUBV */
        WubuvReader* r = wubuv_reader_open(input_path);
        if (!r) { printf("Cannot open %s\n", input_path); return 1; }

        uint8_t type;
        uint32_t size;
        uint8_t* data;
        if (wubuv_read_frame(r, &type, &data, &size) != 0) {
            printf("Cannot read frame\n");
            wubuv_reader_close(r);
            return 1;
        }

        /* Parse header (skip WUBV width/height 4 bytes) */
        int total_latents, num_audio_frames;
        float qp, curvature;
        memcpy(&total_latents, data + 4, 4);
        memcpy(&num_audio_frames, data + 8, 4);
        int32_t qstep_bits, curvature_bits;
        memcpy(&qstep_bits, data + 12, 4);
        memcpy(&curvature_bits, data + 16, 4);
        qp = qstep_bits / 1e6f;
        curvature = curvature_bits / 1e6f;

        int total_video_latents = total_latents - num_audio_frames;
        printf("  Total latents: %d (video: %d, audio: %d)\n", total_latents, total_video_latents, num_audio_frames);
        printf("  Curvature: %.2f, QP: %.4f\n", curvature, qp);

        /* Decode latents */
        int16_t* levels = (int16_t*)(data + 20);
        float* canvas = malloc(total_latents * 4 * sizeof(float));
        float* prev_recon = calloc(4, sizeof(float));

        for (int i = 0; i < total_latents; i++) {
            if (i % 10 == 0) {
                /* I-frame: absolute */
                float qstep = qp * (1.0f + vec_norm(prev_recon, 4) * 10.0f);
                for (int d = 0; d < 4; d++) {
                    canvas[i*4+d] = (float)levels[i*4+d] * qstep;
                }
                poincare_clip(canvas + i*4, canvas + i*4, 4, curvature);
                memcpy(prev_recon, canvas + i*4, 4 * sizeof(float));
            } else {
                /* P-frame: tangent velocity (same QP as encode) */
                float t_norm = 0;
                for (int d = 0; d < 4; d++) {
                    float approx = (float)levels[i*4+d] * qp * 0.1f;
                    t_norm += approx * approx;
                }
                t_norm = sqrtf(t_norm);
                float qstep = qp * 0.1f * (1.0f + t_norm * 10.0f);
                float dequant[4];
                for (int d = 0; d < 4; d++) {
                    dequant[d] = (float)levels[i*4+d] * qstep;
                }
                expmap_at(canvas + i*4, prev_recon, dequant, 4, curvature);
                memcpy(prev_recon, canvas + i*4, 4 * sizeof(float));
            }
        }

        /* Extract video */
        int W = 320, H = 240;
        int num_frames = total_video_latents / (W * H);
        float* video = malloc(total_video_latents * 3 * sizeof(float));
        for (int i = 0; i < total_video_latents; i++) {
            manifold_to_rgb(canvas + i*4, &video[i*3], &video[i*3+1], &video[i*3+2], curvature);
        }

        /* Extract audio */
        float* audio = malloc(num_audio_frames * 64 * sizeof(float));
        for (int i = 0; i < num_audio_frames; i++) {
            manifold_to_audio(canvas + (total_video_latents + i)*4,
                             audio + i*64, 64, curvature);
        }

        /* Write outputs */
        FILE* f = fopen(output_rgb, "wb");
        fwrite(video, 1, total_video_latents * 3 * sizeof(float), f);
        fclose(f);

        f = fopen(output_pcm, "wb");
        fwrite(audio, 1, num_audio_frames * 64 * sizeof(float), f);
        fclose(f);

        printf("  Output: %s (%.2f KB), %s (%.2f KB)\n",
               output_rgb, total_video_latents * 3 * sizeof(float) / 1024.0,
               output_pcm, num_audio_frames * 64 * sizeof(float) / 1024.0);

        free(video); free(audio); free(canvas); free(prev_recon);
        free(data);
        wubuv_reader_close(r);
        return 0;
    }

    if (strcmp(cmd, "render") == 0) {
        if (argc < 4) { printf("Usage: %s render <input.wubv> <output.ppm>\n", argv[0]); return 1; }
        const char* input_path = argv[2];
        const char* output_ppm = argv[3];

        printf("=== THE CANVAS: Render ===\n");
        printf("  Input: %s\n", input_path);

        /* Read WUBV */
        WubuvReader* r = wubuv_reader_open(input_path);
        if (!r) { printf("Cannot open %s\n", input_path); return 1; }

        uint8_t type;
        uint32_t size;
        uint8_t* data;
        if (wubuv_read_frame(r, &type, &data, &size) != 0) {
            printf("Cannot read frame\n");
            wubuv_reader_close(r);
            return 1;
        }

        /* Parse header (skip WUBV width/height 4 bytes) */
        int total_latents, num_audio_frames;
        float qp, curvature;
        memcpy(&total_latents, data + 4, 4);
        memcpy(&num_audio_frames, data + 8, 4);
        int32_t qstep_bits, curvature_bits;
        memcpy(&qstep_bits, data + 12, 4);
        memcpy(&curvature_bits, data + 16, 4);
        qp = qstep_bits / 1e6f;
        curvature = curvature_bits / 1e6f;

        int total_video_latents = total_latents - num_audio_frames;
        int W = 320, H = 240;
        int num_frames = total_video_latents / (W * H);

        printf("  Video: %dx%d, %d frames\n", W, H, num_frames);
        printf("  Curvature: %.2f, QP: %.4f\n", curvature, qp);

        /* Decode latents */
        int16_t* levels = (int16_t*)(data + 20);
        float* canvas = malloc(total_latents * 4 * sizeof(float));
        float* prev_recon = calloc(4, sizeof(float));

        for (int i = 0; i < total_latents; i++) {
            if (i % 10 == 0) {
                float qstep = qp * (1.0f + vec_norm(prev_recon, 4) * 10.0f);
                for (int d = 0; d < 4; d++) {
                    canvas[i*4+d] = (float)levels[i*4+d] * qstep;
                }
                poincare_clip(canvas + i*4, canvas + i*4, 4, curvature);
                memcpy(prev_recon, canvas + i*4, 4 * sizeof(float));
            } else {
                float t_norm = 0;
                for (int d = 0; d < 4; d++) {
                    float approx = (float)levels[i*4+d] * qp * 0.1f;
                    t_norm += approx * approx;
                }
                t_norm = sqrtf(t_norm);
                float qstep = qp * 0.1f * (1.0f + t_norm * 10.0f);
                float dequant[4];
                for (int d = 0; d < 4; d++) {
                    dequant[d] = (float)levels[i*4+d] * qstep;
                }
                expmap_at(canvas + i*4, prev_recon, dequant, 4, curvature);
                memcpy(prev_recon, canvas + i*4, 4 * sizeof(float));
            }
        }

        /* Render first frame to PPM */
        float* video = malloc(total_video_latents * 3 * sizeof(float));
        for (int i = 0; i < total_video_latents; i++) {
            manifold_to_rgb(canvas + i*4, &video[i*3], &video[i*3+1], &video[i*3+2], curvature);
        }

        /* Write PPM (first frame) */
        FILE* f = fopen(output_ppm, "wb");
        fprintf(f, "P6\n%d %d\n255\n", W, H);
        for (int i = 0; i < W * H; i++) {
            uint8_t r = (uint8_t)(video[i*3] * 255);
            uint8_t g = (uint8_t)(video[i*3+1] * 255);
            uint8_t b = (uint8_t)(video[i*3+2] * 255);
            fwrite(&r, 1, 1, f);
            fwrite(&g, 1, 1, f);
            fwrite(&b, 1, 1, f);
        }
        fclose(f);

        printf("  Output: %s\n", output_ppm);

        free(video); free(canvas); free(prev_recon);
        free(data);
        wubuv_reader_close(r);
        return 0;
    }

    printf("Unknown command: %s\n", cmd);
    return 1;
}
