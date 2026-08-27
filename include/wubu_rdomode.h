/* GAP-C029: rate-distortion mode decision */
#ifndef WUBU_RDOMODE_H
#define WUBU_RDOMODE_H
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
    WUBU_RD_SKIP=0,
    WUBU_RD_RESIDUAL=1
} WubuRDModeTag;

typedef struct {
    WubuRDModeTag mode;
    float cost;       /* J = D + lambda*R */
    float distortion;
    int bits;
} WubuRDMode;

float wubu_rd_geodesic(const float* a,const float* b,int D,float c);
float wubu_rd_lambda(int qp);
WubuRDMode wubu_rd_decide(const float* orig,const float* pred,
                           const float* recon_residual,
                           int D,float c,float lambda,
                           float rate_skip,float rate_residual);
#ifdef __cplusplus
}
#endif
#endif
