#include "wubu_idct8x8.h"
#ifndef M_PI
#define M_PI 3.14159265358979
#endif
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DCT_N 8

static double dct_alpha(int u){
    return u==0 ? sqrt(1.0/DCT_N) : sqrt(2.0/DCT_N);
}

static void fdct_1d(int* x){
    double tmp[DCT_N];
    for(int u=0;u<DCT_N;u++){
        double sum=0;
        for(int n=0;n<DCT_N;n++)
            sum+=dct_alpha(u)*cos((2*n+1)*u*M_PI/(2*DCT_N))*x[n];
        tmp[u]=sum;
    }
    for(int u=0;u<DCT_N;u++)x[u]=(int)lround(tmp[u]);
}

static void idct_1d(int* x){
    double tmp[DCT_N];
    for(int n=0;n<DCT_N;n++){
        double sum=0;
        for(int u=0;u<DCT_N;u++)
            sum+=dct_alpha(u)*cos((2*n+1)*u*M_PI/(2*DCT_N))*x[u];
        tmp[n]=sum;
    }
    for(int n=0;n<DCT_N;n++)x[n]=(int)lround(tmp[n]);
}

void wubu_dct8x8_forward(const int* input,int* output){
    int tmp[64];
    memcpy(tmp,input,sizeof(int)*64);
    for(int r=0;r<8;r++)fdct_1d(tmp+r*8);
    for(int c=0;c<8;c++){
        int col[8];
        for(int r=0;r<8;r++)col[r]=tmp[r*8+c];
        fdct_1d(col);
        for(int r=0;r<8;r++)output[r*8+c]=col[r];
    }
}

void wubu_dct8x8_inverse(const int* input,int* output){
    int tmp[64];
    memcpy(tmp,input,sizeof(int)*64);
    for(int c=0;c<8;c++){
        int col[8];
        for(int r=0;r<8;r++)col[r]=tmp[r*8+c];
        idct_1d(col);
        for(int r=0;r<8;r++)tmp[r*8+c]=col[r];
    }
    for(int r=0;r<8;r++)
        idct_1d(tmp+r*8);
    memcpy(output,tmp,sizeof(int)*64);
}

static const int qmat_base[64]={
    16,11,10,16,24,40,51,61,
    12,12,14,19,26,58,60,55,
    14,13,16,24,40,57,69,56,
    14,17,22,29,51,87,80,62,
    18,22,37,56,68,109,103,77,
    24,35,55,64,81,104,113,92,
    49,64,78,87,103,121,120,101,
    72,92,95,98,112,100,103,99
};

void wubu_dct8x8_quantize(const int* coeffs,int quality,int* quantized){
    float scale=(quality==0)?1.0f:(float)(1+quality*2);
    for(int i=0;i<64;i++){
        int qstep=(int)(qmat_base[i]*scale/16.0f);
        if(qstep<1)qstep=1;
        int v=coeffs[i];
        quantized[i]=(v>=0)?(v+qstep/2)/qstep:(v-qstep/2)/qstep;
    }
}

void wubu_dct8x8_dequantize(const int* quantized,int quality,int* coeffs){
    float scale=(quality==0)?1.0f:(float)(1+quality*2);
    for(int i=0;i<64;i++){
        int qstep=(int)(qmat_base[i]*scale/16.0f);
        if(qstep<1)qstep=1;
        coeffs[i]=quantized[i]*qstep;
    }
}
