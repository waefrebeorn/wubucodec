/*
 * wubu_nn.h -- C11 Neural Network Layer Library
 *
 * Slermed from PyTorch nn.Module classes used in bytropix Python code.
 * Each layer maps 1:1 to its Python equivalent:
 *
 *   Python              | C11
 *   --------------------|------------------------
 *   nn.Linear(in,out)   | wubu_linear_forward(W, x, b)
 *   nn.LayerNorm(dim)   | wubu_layer_norm(x, gamma, beta)
 *   nn.GELU()           | wubu_gelu(x)
 *   nn.Tanh()           | wubu_tanh(x)
 *   nn.SiLU()           | wubu_silu(x)
 *   nn.LeakyReLU()      | wubu_leaky_relu(x, alpha)
 *   nn.Dropout(p)       | wubu_dropout(x, p, rng)
 *   nn.Embedding(n,d)   | wubu_embedding_forward(table, idx)
 *   nn.CrossEntropyLoss | wubu_cross_entropy(logits, target)
 *   nn.MSELoss          | wubu_mse_loss(pred, target)
 *   nn.Conv2d            | wubu_conv2d(x, weight, bias, stride, padding)
 *   nn.ConvTranspose2d  | wubu_conv_transpose2d(x, weight, bias, stride)
 *   nn.Sequential        | wubu_sequential_forward(layers, x)
 *   nn.Softmax           | wubu_softmax(x, dim)
 *   nn.Bilinear         | wubu_bilinear(x1, x2, W, b)
 *   nn.GroupNorm        | wubu_group_norm(x, gamma, beta, num_groups)
 *   nn.InstanceNorm2d   | wubu_instance_norm2d(x, gamma, beta)
 *   nn.AdaptiveAvgPool  | wubu_adaptive_avg_pool2d(x, out_h, out_w)
 *   nn.MultiheadAttention| wubu_multihead_attention(q, k, v, W_q, W_k, W_v, W_o, mask)
 *   nn.BCEWithLogitsLoss| wubu_bce_with_logits(pred, target)
 *
 * All functions operate on flat float arrays in NCHW or [B, ...] layout.
 * Output buffers must be pre-allocated by the caller.
 *
 * No external dependencies beyond wubumath.h (C11 standard library + math).
 */

#ifndef WUBU_NN_H
#define WUBU_NN_H

#include "wubumath.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ===================================================================
 * Linear Layer (nn.Linear)
 * y = x @ W^T + b
 * =================================================================== */

typedef struct {
    float* weight;   /* [out_features, in_features] */
    float* bias;     /* [out_features] (nullable) */
    int in_features;
    int out_features;
} WubuLinear;

void wubu_linear_init(WubuLinear* layer, int in_features, int out_features, int use_bias);
void wubu_linear_free(WubuLinear* layer);
void wubu_linear_forward(const WubuLinear* layer, const float* x, float* output, int B);
/* x: [B, in_features], output: [B, out_features] */

/* ===================================================================
 * Layer Normalization (nn.LayerNorm)
 * y = (x - mean) / sqrt(var + eps) * gamma + beta
 * =================================================================== */

typedef struct {
    float* weight;   /* gamma [normalized_shape] */
    float* bias;     /* beta [normalized_shape] */
    int normalized_shape;
    float eps;
} WubuLayerNorm;

void wubu_layer_norm_init(WubuLayerNorm* layer, int normalized_shape, float eps);
void wubu_layer_norm_free(WubuLayerNorm* layer);
void wubu_layer_norm_forward(const WubuLayerNorm* layer, const float* x, float* output, int B);
/* x: [B, normalized_shape], output: [B, normalized_shape] */

/* sigmoid — used by tests and other layers */
float sigmoid(float x);

/* ===================================================================
 * Activation Functions
 * =================================================================== */

void wubu_gelu_forward(const float* x, float* output, int n);
/* GELU: x * Φ(x) = x * 0.5 * (1 + erf(x / sqrt(2))) */

void wubu_tanh_forward(const float* x, float* output, int n);

void wubu_silu_forward(const float* x, float* output, int n);
/* SiLU: x * sigmoid(x) */

void wubu_leaky_relu_forward(const float* x, float* output, int n, float alpha);
/* LeakyReLU: x if x >= 0, alpha*x otherwise */

void wubu_relu_forward(const float* x, float* output, int n);

/* ===================================================================
 * Dropout (nn.Dropout)
 * x_out = x * mask / (1 - p) during training
 * =================================================================== */

void wubu_dropout_forward(const float* x, float* output, int n, float p, WubuRNG* rng);
/* Scales by 1/(1-p) to maintain expected value (inverted dropout) */

void wubu_dropout_mask_free(float* mask);

/* ===================================================================
 * Embedding (nn.Embedding)
 * output = weight[indices]
 * =================================================================== */

typedef struct {
    float* weight;   /* [num_embeddings, embedding_dim] */
    int num_embeddings;
    int embedding_dim;
} WubuEmbedding;

void wubu_embedding_init(WubuEmbedding* emb, int num_embeddings, int embedding_dim);
void wubu_embedding_free(WubuEmbedding* emb);
void wubu_embedding_forward(const WubuEmbedding* emb, const int* indices, float* output, int batch_size);
/* indices: [batch_size], output: [batch_size, embedding_dim] */

/* ===================================================================
 * Convolution (nn.Conv2d)
 * Slermed from torch.nn.Conv2d with kernel_size=3, stride=1, padding=1
 * =================================================================== */

typedef struct {
    float* weight;   /* [out_channels, in_channels, kH, kW] */
    float* bias;     /* [out_channels] (nullable) */
    int in_channels;
    int out_channels;
    int kernel_size; /* square kernel (kH == kW) */
    int stride;
    int padding;
    int dilation;
} WubuConv2d;

void wubu_conv2d_init(WubuConv2d* conv, int in_channels, int out_channels,
                       int kernel_size, int stride, int padding, int use_bias);
void wubu_conv2d_free(WubuConv2d* conv);
void wubu_conv2d_forward(const WubuConv2d* conv, const float* x,
                         int B, int H, int W_img, float* output);
/* x: [B, in_channels, H, W_img], output: [B, out_channels, out_H, out_W] */

/* ===================================================================
 * Convolution Transpose (nn.ConvTranspose2d)
 * =================================================================== */

typedef struct {
    float* weight;   /* [in_channels, out_channels, kH, kW] */
    float* bias;     /* [out_channels] (nullable) */
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int output_padding;
} WubuConvTranspose2d;

void wubu_conv_transpose2d_init(WubuConvTranspose2d* conv, int in_channels, int out_channels,
                                  int kernel_size, int stride, int padding, int output_padding, int use_bias);
void wubu_conv_transpose2d_free(WubuConvTranspose2d* conv);
void wubu_conv_transpose2d_forward(const WubuConvTranspose2d* conv, const float* x,
                                    int B, int H, int W_in, float* output);

/* ===================================================================
 * Group Normalization (nn.GroupNorm)
 * =================================================================== */

typedef struct {
    float* weight;   /* gamma [num_channels] */
    float* bias;     /* beta [num_channels] */
    int num_groups;
    int num_channels;
    float eps;
} WubuGroupNorm;

void wubu_group_norm_init(WubuGroupNorm* gn, int num_groups, int num_channels, float eps);
void wubu_group_norm_free(WubuGroupNorm* gn);
void wubu_group_norm_forward(const WubuGroupNorm* gn, const float* x,
                              int B, int C, int H, int W_img, float* output);

/* ===================================================================
 * Instance Normalization 2D (nn.InstanceNorm2d)
 * =================================================================== */

typedef struct {
    float* weight;   /* gamma [C] */
    float* bias;     /* beta [C] */
    int num_channels;
    float eps;
} WubuInstanceNorm2d;

void wubu_instance_norm2d_init(WubuInstanceNorm2d* in, int num_channels, float eps);
void wubu_instance_norm2d_free(WubuInstanceNorm2d* in);
void wubu_instance_norm2d_forward(const WubuInstanceNorm2d* in, const float* x,
                                  int B, int C, int H, int W_img, float* output);

/* ===================================================================
 * Adaptive Average Pooling (nn.AdaptiveAvgPool2d)
 * =================================================================== */

void wubu_adaptive_avg_pool2d(const float* x, int B, int C, int H, int W_img,
                                int out_h, int out_w, float* output);
/* output: [B, C, out_h, out_w] */

/* ===================================================================
 * Softmax (nn.Softmax)
 * =================================================================== */

void wubu_softmax(const float* x, float* output, int n);

/* ===================================================================
 * Bilinear Layer (nn.Bilinear)
 * y = x1 @ W @ x2 + b
 * =================================================================== */

typedef struct {
    float* weight;   /* [out_features, in1_features, in2_features] */
    float* bias;     /* [out_features] (nullable) */
    int in1_features;
    int in2_features;
    int out_features;
} WubuBilinear;

void wubu_bilinear_init(WubuBilinear* bl, int in1, int in2, int out, int use_bias);
void wubu_bilinear_free(WubuBilinear* bl);
void wubu_bilinear_forward(const WubuBilinear* bl, const float* x1, const float* x2,
                            int B, float* output);
/* x1: [B, in1], x2: [B, in2], output: [B, out] */

/* ===================================================================
 * Loss Functions
 * =================================================================== */

float wubu_cross_entropy_loss(const float* logits, const int* targets, int batch_size, int num_classes);
/* logits: [B, num_classes] (raw scores), targets: [B] (class indices) */

float wubu_mse_loss(const float* pred, const float* target, int n);
/* Mean squared error averaged over all elements */

float wubu_bce_with_logits_loss(const float* pred, const float* target, int n);
/* Binary cross-entropy with logits: -[t*log(sigmoid(p)) + (1-t)*log(1-sigmoid(p))] */

/* ===================================================================
 * Multi-Head Attention (nn.MultiheadAttention)
 * Slermed from torch.nn.MultiheadAttention
 * =================================================================== */

typedef struct {
    float* W_q;      /* [d_model, d_model] */
    float* W_k;      /* [d_model, d_model] */
    float* W_v;      /* [d_model, d_model] */
    float* W_o;      /* [d_model, d_model] */
    float* b_q;      /* [d_model] */
    float* b_k;      /* [d_model] */
    float* b_v;      /* [d_model] */
    float* b_o;      /* [d_model] */
    int d_model;
    int num_heads;
    int use_bias;
} WubuMultiheadAttention;

void wubu_mha_init(WubuMultiheadAttention* mha, int d_model, int num_heads, int use_bias);
void wubu_mha_free(WubuMultiheadAttention* mha);
void wubu_mha_forward(WubuMultiheadAttention* mha, const float* Q, const float* K, const float* V,
                       int B, int seq_len, const float* mask, float* output);
/* Q, K, V: [B, seq_len, d_model], output: [B, seq_len, d_model] */
/* mask: [B, seq_len, seq_len] or NULL (causal mask if provided) */

/* ===================================================================
 * Sequential Container (nn.Sequential)
 * =================================================================== */

typedef struct WubuLayer {
    void (*forward)(struct WubuLayer* self, const float* input, float* output);
    void* params;
    int in_size;
    int out_size;
    struct WubuLayer* next;
} WubuLayer;

typedef struct {
    WubuLayer* first;
    WubuLayer* last;
    int num_layers;
} WubuSequential;

void wubu_sequential_init(WubuSequential* seq);
void wubu_sequential_add(WubuSequential* seq, WubuLayer* layer);
void wubu_sequential_forward(WubuSequential* seq, const float* input, float* output);
void wubu_sequential_free(WubuSequential* seq);

/* ===================================================================
 * Gradient Clipping (torch.nn.utils.clip_grad_norm_)
 * =================================================================== */

void wubu_clip_grad_norm(float* gradients, int n, float max_norm);
/* Clips gradient norm to max_norm. gradients: [total_elements] */

/* ===================================================================
 * Utility: Parameter Initialization
 * =================================================================== */

void wubu_init_xavier_uniform(float* weight, int n, int fan_in, int fan_out);
void wubu_init_zeros(float* buf, int n);
void wubu_init_ones(float* buf, int n);

#endif /* WUBU_NN_H */
