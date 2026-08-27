/*
 * wubu_transform.c -- GROUP 7: Variable-size transforms
 *
 * G7.01: 4×4 integer DCT (H.264 core)
 * G7.02: 16×16 integer DCT
 * G7.03: 32×32 integer DCT (HEVC/VVC)
 * G7.05: Transform skip mode (identity)
 * G7.09: Quantization matrices per size/type
 */
#define M_PI 3.14159265358979f
#include "wubu_transform.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== Generic N×N float DCT via matrix multiplication ===== */

/* precompute DCT basis matrix for given size */
static void dct_basis(int n,double* mat){
    /* C(u) = c(u) * cos((2x+1)*u*pi/(2n)) where c(0)=sqrt(1/n), c(u)=sqrt(2/n) */
    for(int u=0;u<n;u++){
        double cu=(u==0)?sqrt(1.0/n):sqrt(2.0/n);
        for(int x=0;x<n;x++)
            mat[u*n+x]=cu*cos((2*x+1)*u*M_PI/(2*n));
    }
}

/* forward transform: coeff = basis × input × basis^T (separable) */
static void dct_forward_n(const double* basis,const int16_t* input,
                           int16_t* output,int n){
    double tmp[n*n];
    
    /* rows: tmp = input × B^T (i.e., each row of input transformed) */
    for(int y=0;y<n;y++)
        for(int u=0;u<n;u++){
            double sum=0;
            for(int x=0;x<n;x++)
                sum+=input[y*n+x]*basis[u*n+x];
            tmp[y*n+u]=sum;
        }
    
    /* columns: output = B × tmp */
    for(int u=0;u<n;u++)
        for(int v=0;v<n;v++){
            double sum=0;
            for(int y=0;y<n;y++)
                sum+=basis[u*n+y]*tmp[y*n+v];
            output[v*n+u]=(int16_t)lround(sum);
        }
}

/* inverse transform: output = B^T × coeff × B */
static void dct_inverse_n(const double* basis,const int16_t* coeff,
                            int16_t* output,int n){
    double tmp[n*n];
    
    /* tmp = B^T × coeff */
    for(int y=0;y<n;y++)
        for(int x=0;x<n;x++){
            double sum=0;
            for(int u=0;u<n;u++)
                sum+=basis[u*n+y]*coeff[x*n+u];
            tmp[y*n+x]=sum;
        }
    
    /* output = tmp × B */
    for(int y=0;y<n;y++)
        for(int x=0;x<n;x++){
            double sum=0;
            for(int v=0;v<n;v++)
                sum+=tmp[y*n+v]*basis[v*n+x];
            output[y*n+x]=(int16_t)lround(sum);
        }
}

/* ===== Public API ===== */

void wubu_tr_forward(const int16_t* input,int16_t* output,int size){
    double basis[size*size];
    dct_basis(size,basis);
    dct_forward_n(basis,input,output,size);
}

void wubu_tr_inverse(const int16_t* coeff,int16_t* output,int size){
    double basis[size*size];
    dct_basis(size,basis);
    dct_inverse_n(basis,coeff,output,size);
}

/* ===== Multi-size transform support (C9: variable-size transforms) ===== */

/* DST-VII basis for 4x4 intra blocks (H.266-style) — sharper for directional residuals */
static void dst7_basis_4x4(double* mat){
    double n=4.0;
    for(int u=0;u<4;u++)
        for(int v=0;v<4;v++)
            mat[u*4+v]=sqrt(2.0/n)*sin((2*(double)u+1.0)*((double)v+0.5)*M_PI/(2.0*n));
}

/* 4x4 DST-VII forward/inverse */
static void dst7_forward_4x4(const double* basis,const int16_t* input,int16_t* output){
    double tmp[16];
    for(int y=0;y<4;y++)
        for(int u=0;u<4;u++){double sum=0;for(int x=0;x<4;x++)sum+=input[y*4+x]*basis[u*4+x];tmp[y*4+u]=sum;}
    for(int u=0;u<4;u++)
        for(int v=0;v<4;v++){double sum=0;for(int y=0;y<4;y++)sum+=basis[u*4+y]*tmp[y*4+v];output[v*4+u]=(int16_t)lround(sum);}
}
static void dst7_inverse_4x4(const double* basis,const int16_t* coeff,int16_t* output){
    double tmp[16];
    for(int y=0;y<4;y++)
        for(int x=0;x<4;x++){double sum=0;for(int u=0;u<4;u++)sum+=basis[u*4+y]*coeff[x*4+u];tmp[y*4+x]=sum;}
    for(int y=0;y<4;y++)
        for(int x=0;x<4;x++){double sum=0;for(int v=0;v<4;v++)sum+=tmp[y*4+v]*basis[v*4+x];output[y*4+x]=(int16_t)lround(sum);}
}

/* 4x4 DCT — small block for high-frequency/detail regions */
static void dct4_forward(const int16_t* input,int16_t* output){
    static double basis4[16]; static int b4r=0;
    if(!b4r){dct_basis(4,basis4);b4r=1;}
    dct_forward_n(basis4,input,output,4);
}
static void dct4_inverse(const int16_t* coeff,int16_t* output){
    static double basis4[16]; static int b4r=0;
    if(!b4r){dct_basis(4,basis4);b4r=1;}
    dct_inverse_n(basis4,coeff,output,4);
}

/* 16x16 DCT — large block for smooth regions */
static void dct16_forward(const int16_t* input,int16_t* output){
    static double basis16[256]; static int b16r=0;
    if(!b16r){dct_basis(16,basis16);b16r=1;}
    dct_forward_n(basis16,input,output,16);
}
static void dct16_inverse(const int16_t* coeff,int16_t* output){
    static double basis16[256]; static int b16r=0;
    if(!b16r){dct_basis(16,basis16);b16r=1;}
    dct_inverse_n(basis16,coeff,output,16);
}

/* 32x32 DCT — very large block for very smooth regions (HEVC/VVC 32x32) */
static void dct32_forward(const int16_t* input,int16_t* output){
    static double basis32[1024]; static int b32r=0;
    if(!b32r){dct_basis(32,basis32);b32r=1;}
    dct_forward_n(basis32,input,output,32);
}
static void dct32_inverse(const int16_t* coeff,int16_t* output){
    static double basis32[1024]; static int b32r=0;
    if(!b32r){dct_basis(32,basis32);b32r=1;}
    dct_inverse_n(basis32,coeff,output,32);
}

/* transform skip: just copy */
void wubu_tr_skip(const int16_t* input,int16_t* output,int size){
    memcpy(output,input,sizeof(int16_t)*(size_t)(size*size));
}

/* quantize with scaling matrix (per-coefficient weights) */
void wubu_tr_quantize_m(const int16_t* coeffs,const uint8_t* qmat,
                          int16_t* output,int size,int qp){
    int shift=(qp/6)+4;
    int offset=1<<(shift-1);
    for(long i=0;i<(long)size*size;i++){
        int scaled=coeffs[i]*(int)qmat[i];
        output[i]=(int16_t)((scaled+offset)>>shift);
    }
}

void wubu_tr_dequantize_m(const int16_t* quantized,const uint8_t* qmat,
                            int16_t* output,int size,int qp){
    int shift=qp/6+4;
    for(long i=0;i<(long)size*size;i++)
        output[i]=(int16_t)(quantized[i]*qmat[i]<<shift>>4);
}

/* default flat quantization matrix */
static const uint8_t qmat_flat[64]={
    16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16,
    16,16,16,16,16,16,16,16
};

/* JPEG-style quality-aware quantization matrix for 8x8 */
static const uint8_t qmat_8x8[64]={
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68,109,103, 77,
    24, 35, 55, 64, 81,104,113, 92,
    49, 64, 78, 87,103,121,120,101,
    72, 92, 95, 98,112,100,103, 99
};

const uint8_t* wubu_tr_get_qmat(int size,int intra){
    if(size==8)return qmat_8x8;
    return qmat_flat;
}

/* ===== C9: multi-size public API ===== */

int wubu_tr_get_sizes(int* sizes,int max){
    static const int avail[]={4,8,16,32};
    int n=0;
    for(int i=0;i<4 && n<max;i++) sizes[n++]=avail[i];
    return n;
}

int wubu_tr_forward_size(int size,const int16_t* input,int16_t* output){
    switch(size){
        case 4: dct4_forward(input,output); return 0;
        case 8: wubu_tr_forward(input,output,8); return 0;
        case 16: dct16_forward(input,output); return 0;
        case 32: dct32_forward(input,output); return 0;
        default: return -1;
    }
}

int wubu_tr_inverse_size(int size,const int16_t* coeff,int16_t* output){
    switch(size){
        case 4: dct4_inverse(coeff,output); return 0;
        case 8: wubu_tr_inverse(coeff,output,8); return 0;
        case 16: dct16_inverse(coeff,output); return 0;
        case 32: dct32_inverse(coeff,output); return 0;
        default: return -1;
    }
}

int wubu_tr_forward_dst7_4x4(const int16_t* input,int16_t* output){
    static double basis4[16]; static int b4r=0;
    if(!b4r){dst7_basis_4x4(basis4);b4r=1;}
    dst7_forward_4x4(basis4,input,output);
    return 0;
}

int wubu_tr_inverse_dst7_4x4(const int16_t* coeff,int16_t* output){
    static double basis4[16]; static int b4r=0;
    if(!b4r){dst7_basis_4x4(basis4);b4r=1;}
    dst7_inverse_4x4(basis4,coeff,output);
    return 0;
}

/* JPEG-style quality-aware quantization matrix for 4x4 */
static const uint8_t qmat_4x4[16]={
    16, 11, 10, 16,
    12, 12, 14, 19,
    14, 13, 16, 24,
    18, 22, 37, 56
};

/* JPEG-style quantization matrix for 16x16 */
static const uint8_t qmat_16x16[256]={
/* Z-pattern for 16x16: zigzag pattern */
16,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,
11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,
12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,
13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,
14,15,16,17,24,25,26,27,28,30,31,32,33,34,35,36,
15,16,17,24,25,30,31,32,33,34,35,36,37,38,39,40,
16,17,24,25,30,32,35,36,37,38,39,40,41,42,43,44,
17,18,24,26,32,35,40,42,43,44,45,46,47,48,49,50,
18,19,25,27,33,36,42,45,46,47,48,49,50,51,52,53,
19,20,26,28,34,37,43,46,47,48,49,50,51,52,53,54,
20,21,27,29,35,38,44,47,48,49,50,51,52,53,54,55,
21,22,28,30,36,39,45,48,49,50,51,52,53,54,55,56,
22,23,29,31,37,40,46,49,50,51,52,53,54,55,56,57,
23,24,30,32,38,41,47,50,51,52,53,54,55,56,57,58,
24,25,31,33,39,42,48,51,52,53,54,55,56,57,58,59
};

/* 32x32 flat quantization matrix (mutable, initialized on first use) */
static uint8_t qmat_32x32[1024];
static int qmat_32x32_ready=0;
static void init_qmat_32x32(void){
    if(qmat_32x32_ready)return;
    for(int i=0;i<1024;i++) qmat_32x32[i]=16;
    qmat_32x32_ready=1;
}

/* Get quantization matrix for a given transform size */
const uint8_t* wubu_tr_get_qmat_size(int size,int intra){
    switch(size){
        case 4: return qmat_4x4;
        case 8: return intra?qmat_8x8:qmat_flat;
        case 16: return qmat_16x16;
        case 32: init_qmat_32x32(); return qmat_32x32;
        default: return qmat_flat;
    }
}

/* Map transform size to qmat index for trellis quantization */
int wubu_tr_size_to_index(int size){
    switch(size){
        case 4: return 0;
        case 8: return 1;
        case 16: return 2;
        case 32: return 3;
        default: return -1;
    }
}
