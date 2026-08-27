/*
 * wubu_loss.c -- Loss manifold computation (slermed from vhf_audio.py train_step)
 *
 * Implements the HSL-aware loss function:
 *   composite = LUMA_WEIGHT * L_loss + PHASE_WEIGHT * H_loss + SAT_WEIGHT * S_loss
 *
 * Where:
 *   H_loss = circular_l1_loss(pred_h, gt_h)  -- hue manifold
 *   S_loss = |pred_s - gt_s|                  -- saturation L1
 *   L_loss = |pred_l - gt_l|                  -- lightness L1
 */

#include "wubumath.h"

/* Weights matching vhf_audio.py */
#define LUMA_WEIGHT  10.0f
#define PHASE_WEIGHT  2.0f
#define SAT_WEIGHT    1.0f

WubuLoss wubu_compute_loss(const float* pred_rgb, const float* gt_rgb, int N) {
    WubuLoss loss = {0.0f, 0.0f, 0.0f, 0.0f};

    if (N == 0) return loss;

    float h_sum = 0.0f, s_sum = 0.0f, l_sum = 0.0f;

    for (int i = 0; i < N; ++i) {
        /* Normalize from [-1,1] to [0,1] for HSL conversion */
        WubuRGB pred = {
            .r = fmaxf(0.0f, fminf(1.0f, (pred_rgb[i * 3 + 0] + 1.0f) * 0.5f)),
            .g = fmaxf(0.0f, fminf(1.0f, (pred_rgb[i * 3 + 1] + 1.0f) * 0.5f)),
            .b = fmaxf(0.0f, fminf(1.0f, (pred_rgb[i * 3 + 2] + 1.0f) * 0.5f))
        };
        WubuRGB gt = {
            .r = fmaxf(0.0f, fminf(1.0f, (gt_rgb[i * 3 + 0] + 1.0f) * 0.5f)),
            .g = fmaxf(0.0f, fminf(1.0f, (gt_rgb[i * 3 + 1] + 1.0f) * 0.5f)),
            .b = fmaxf(0.0f, fminf(1.0f, (gt_rgb[i * 3 + 2] + 1.0f) * 0.5f))
        };

        WubuHSL pred_hsl = wubu_rgb_to_hsl(pred);
        WubuHSL gt_hsl = wubu_rgb_to_hsl(gt);

        h_sum += wubu_circular_l1_loss(pred_hsl.h, gt_hsl.h);
        s_sum += fabsf(pred_hsl.s - gt_hsl.s);
        l_sum += fabsf(pred_hsl.l - gt_hsl.l);
    }

    loss.phase_loss = h_sum / (float)N;
    loss.sat_loss = s_sum / (float)N;
    loss.luma_loss = l_sum / (float)N;
    loss.composite_loss = LUMA_WEIGHT * loss.luma_loss +
                         PHASE_WEIGHT * loss.phase_loss +
                         SAT_WEIGHT * loss.sat_loss;

    return loss;
}
