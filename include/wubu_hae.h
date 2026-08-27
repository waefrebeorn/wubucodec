/* GAP-C037: hyperbolic autoencoder */
#ifndef WUBU_HAE_H
#define WUBU_HAE_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int D_in,D_code;float c;
    float* W_enc; /* [D_code, D_in] */
    float* W_dec; /* [D_in, D_code] */
} WubuHAE;

int  wubu_hae_init(WubuHAE* ae,int D_in,int D_code,unsigned seed);
void wubu_hae_free(WubuHAE* ae);
void wubu_hae_encode(const WubuHAE* ae,const float* x,float* z);
void wubu_hae_decode(const WubuHAE* ae,const float* z,float* x_hat);
float wubu_hae_recon_mse(const WubuHAE* ae,const float* xs,int n);
#ifdef __cplusplus
}
#endif
#endif
