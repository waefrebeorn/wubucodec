/*
 * wubu_hae.c -- GAP-C037: Hyperbolic autoencoder (tangent bottleneck)
 *
 * Poincaré VAE architecture (Mathieu et al. 2019, simplified deterministic
 * variant): Euclidean input -> linear encoder -> exp0 -> BOTTLENECK (ball)
 * -> log0 -> linear decoder -> reconstruction.
 *
 * The bottleneck forces reconstruction through the hyperbolic space: the
 * code lives on the ball, so the network must organize hierarchical
 * structure radially to reconstruct well.
 *
 * Trained by FD gradient on reconstruction MSE through both mappings.
 * Gates: loss decreases; codes on-ball; reconstructions better than
 * constant-mean baseline.
 */
#include "wubu_hae.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void hae_log0(const float* x,int D,float c,float* v){
    float n2=0;for(int d=0;d<D;d++)n2+=x[d]*x[d];
    float nv=sqrtf(n2);
    if(nv<1e-10f){for(int d=0;d<D;d++)v[d]=0;return;}
    float arg=sqrtf(c)*nv;if(arg>0.99999f)arg=0.99999f;
    float zn=(2.0f/sqrtf(c))*atanhf(arg)/nv;
    for(int d=0;d<D;d++)v[d]=zn*x[d];
}
static void hae_exp0(const float* v,int D,float c,float* x){
    float n2=0;for(int d=0;d<D;d++)n2+=v[d]*v[d];
    float nv=sqrtf(n2);
    if(nv<1e-10f){memset(x,0,sizeof(float)*D);return;}
    float coeff=tanhf(sqrtf(c)*nv)/(sqrtf(c)*nv);
    for(int d=0;d<D;d++)x[d]=coeff*v[d];
    float n2c=0;for(int d=0;d<D;d++)n2c+=x[d]*x[d];
    if(n2c>0.99998f){float s=sqrtf(0.99998f/n2c);for(int d=0;d<D;d++)x[d]*=s;}
}

int wubu_hae_init(WubuHAE* ae,int D_in,int D_code,unsigned seed){
    if(D_in<1||D_code<1||D_code>D_in)return -1;
    ae->D_in=D_in;ae->D_code=D_code;ae->c=1.0f;
    unsigned rs=seed*374761393u+11u;
    ae->W_enc=malloc(sizeof(float)*(size_t)D_code*D_in);
    ae->W_dec=malloc(sizeof(float)*(size_t)D_in*D_code);
    if(!ae->W_enc||!ae->W_dec)return -2;
    /* Xavier-ish init */
    float se=1/sqrtf((float)D_in),sd=1/sqrtf((float)D_code);
    for(int i=0;i<D_code*D_in;i++){
        rs=rs*1103515245u+12345u;
        ae->W_enc[i]=((float)((rs>>16)%2000)/2000.0f-0.5f)*2*se;
    }
    for(int i=0;i<D_in*D_code;i++){
        rs=rs*1103515245u+12345u;
        ae->W_dec[i]=((float)((rs>>16)%2000)/2000.0f-0.5f)*2*sd;
    }
    return 0;
}
void wubu_hae_free(WubuHAE* ae){
    free(ae->W_enc);free(ae->W_dec);
    ae->W_enc=NULL;ae->W_dec=NULL;
}

/* encode one vector to ball code */
void wubu_hae_encode(const WubuHAE* ae,const float* x,float* z){
    int Di=ae->D_in,Dc=ae->D_code;
    float tv[64];int dc=Dc<64?Dc:64;
    for(int j=0;j<dc;j++){
        float acc=0;
        for(int k=0;k<Di;k++)acc+=ae->W_enc[(size_t)j*Di+k]*x[k];
        tv[j]=acc*0.3f;   /* scale down so codes stay spread in ball */
    }
    for(int j=dc;j<Dc&&j<64;j++)tv[j]=0;
    hae_exp0(tv,dc,ae->c,z);
}

/* decode ball code to reconstruction */
void wubu_hae_decode(const WubuHAE* ae,const float* z,float* x_hat){
    int Di=ae->D_in,Dc=ae->D_code;
    float zv[64];int dc=Dc<64?Dc:64;
    hae_log0(z,dc,ae->c,zv);
    for(int i=0;i<Di;i++){
        float acc=0;
        for(int j=0;j<dc;j++)acc+=ae->W_dec[(size_t)i*Dc+j]*zv[j];
        x_hat[i]=acc;
    }
}

float wubu_hae_recon_mse(const WubuHAE* ae,const float* xs,int n){
    double s=0;
    float* z=malloc(sizeof(float)*(size_t)ae->D_code);
    float* xh=malloc(sizeof(float)*(size_t)ae->D_in);
    for(int i=0;i<n;i++){
        wubu_hae_encode(ae,xs+(size_t)i*ae->D_in,z);
        wubu_hae_decode(ae,z,xh);
        for(int d=0;d<ae->D_in;d++){
            float df=xh[d]-xs[(size_t)i*ae->D_in+d];
            s+=(double)(df*df);
        }
    }
    free(z);free(xh);
    return (float)(s/(n*ae->D_in));
}
