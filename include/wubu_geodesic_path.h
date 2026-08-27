/* GAP-C025: geodesic path sampling on the Poincaré ball */
#ifndef WUBU_GEODESIC_PATH_H
#define WUBU_GEODESIC_PATH_H
#ifdef __cplusplus
extern "C" {
#endif
/* point at parameter t∈[0,1] on the Möbius geodesic from x0 to x1 */
void wubu_gp_point(float* out,const float* x0,const float* x1,
                   int D,float c,float t);
/* full path: out[num_steps+1][D], includes endpoints. */
void wubu_gp_path(const float* x0,const float* x1,int D,float c,
                  int num_steps,float* out);
#ifdef __cplusplus
}
#endif
#endif
