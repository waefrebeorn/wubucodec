/* GAP-C021: Hyperbolic Rotary Positional Encoding */
#ifndef WUBU_HOPE_H
#define WUBU_HOPE_H
#ifdef __cplusplus
extern "C" {
#endif
/* encode position `pos` onto feature pairs using hyperbolic sin/cos rotation.
 * out[D] = boosted x[D]. base controls frequency spectrum (typically 10000). */
void wubu_hope_encode(float* out,const float* x,int D,int pos,float base);
/* batch wrapper: [B,T,D] in-place style to out[B*T*D] */
void wubu_hope_encode_batch(float* out,const float* x,int B,int T,int D,float base);
#ifdef __cplusplus
}
#endif
#endif
