/*
 * wubu_q_controller.c -- Q-Controller (slermed from vhf_audio.py)
 *
 * Adaptive learning rate controller with warmup and exploration.
 * Uses Q-table to select learning rate change factors based on
 * recent metric history.
 */

#include "wubumath.h"
#include <stdlib.h>
#include <string.h>

const QControllerConfig WUBU_Q_CONTROLLER_DEFAULT = {
    .num_lr_actions = 5,
    .lr_change_factors = {0.9f, 0.95f, 1.0f, 1.05f, 1.1f},
    .learning_rate_q = 0.1f,
    .lr_min = 1e-5f,
    .lr_max = 1e-2f,
    .metric_history_len = 100,
    .exploration_rate_q = 0.3f,
    .min_exploration_rate = 0.05f,
    .exploration_decay = 0.9998f,
    .warmup_steps = 500,
    .warmup_lr_start = 1e-6f
};

void wubu_q_controller_init(QController* qc, const QControllerConfig* config) {
    qc->q_table = (float*)calloc((size_t)config->num_lr_actions, sizeof(float));
    qc->metric_history = (float*)calloc((size_t)config->metric_history_len, sizeof(float));
    qc->current_lr = config->warmup_lr_start;
    qc->exploration_rate = config->exploration_rate_q;
    qc->step_count = 0;
    qc->last_action_idx = -1;
    qc->status_code = 0;
}

void wubu_q_controller_choose_action(QController* qc, WubuRNG* rng,
                                        const QControllerConfig* config, float target_lr) {
    if (qc->step_count < config->warmup_steps) {
        /* Warmup phase: linearly interpolate from warmup_lr_start to target_lr */
        float alpha = (float)qc->step_count / (float)config->warmup_steps;
        qc->current_lr = config->warmup_lr_start * (1.0f - alpha) + target_lr * alpha;
        qc->step_count++;
        qc->status_code = 0;
        qc->last_action_idx = -1;
    } else {
        /* Regular phase: epsilon-greedy Q-table selection */
        float explore = wubu_rng_uniform(rng, 0.0f, 1.0f);
        int action_idx;
        if (explore < qc->exploration_rate) {
            action_idx = wubu_rng_randint(rng, 0, config->num_lr_actions);
        } else {
            /* Argmax Q-table */
            action_idx = 0;
            float best_q = qc->q_table[0];
            for (int i = 1; i < config->num_lr_actions; ++i) {
                if (qc->q_table[i] > best_q) {
                    best_q = qc->q_table[i];
                    action_idx = i;
                }
            }
        }
        float factor = config->lr_change_factors[action_idx];
        qc->current_lr = qc->current_lr * factor;
        if (qc->current_lr < config->lr_min) qc->current_lr = config->lr_min;
        if (qc->current_lr > config->lr_max) qc->current_lr = config->lr_max;
        qc->step_count++;
        qc->last_action_idx = action_idx;
        qc->status_code = 0;
    }
}

void wubu_q_controller_update(QController* qc, float metric_value,
                               const QControllerConfig* config) {
    /* Roll metric history and append new value */
    if (config->metric_history_len > 1) {
        memmove(qc->metric_history, qc->metric_history + 1,
                (size_t)(config->metric_history_len - 1) * sizeof(float));
    }
    qc->metric_history[config->metric_history_len - 1] = metric_value;

    /* Q-update only after warmup and if we have a valid last action */
    if (qc->step_count <= config->warmup_steps || qc->last_action_idx < 0) return;

    /* Compute reward: negative mean of last 10 metrics */
    int start = config->metric_history_len - 10;
    if (start < 0) start = 0;
    float recent_sum = 0.0f;
    int recent_count = 0;
    for (int i = start; i < config->metric_history_len; ++i) {
        recent_sum += qc->metric_history[i];
        recent_count++;
    }
    float reward = -recent_sum / (float)recent_count;

    /* Check if improving (reward > -mean of previous 10) */
    int prev_start = config->metric_history_len - 20;
    if (prev_start < 0) prev_start = 0;
    float prev_sum = 0.0f;
    int prev_count = 0;
    for (int i = prev_start; i < config->metric_history_len - 10; ++i) {
        prev_sum += qc->metric_history[i];
        prev_count++;
    }
    float prev_mean = (prev_count > 0) ? prev_sum / (float)prev_count : 0.0f;
    int is_improving = (reward > -prev_mean) ? 1 : 2;

    /* Q-table update: Q(s,a) += lr * (reward - Q(s,a)) */
    float old_q = qc->q_table[qc->last_action_idx];
    qc->q_table[qc->last_action_idx] = old_q + config->learning_rate_q * (reward - old_q);

    /* Decay exploration rate */
    qc->exploration_rate *= config->exploration_decay;
    if (qc->exploration_rate < config->min_exploration_rate)
        qc->exploration_rate = config->min_exploration_rate;

    qc->status_code = is_improving;
}

void wubu_q_controller_free(QController* qc) {
    if (qc->q_table) { free(qc->q_table); qc->q_table = NULL; }
    if (qc->metric_history) { free(qc->metric_history); qc->metric_history = NULL; }
}
