/* GAP-C030: Poincaré flow matching (GGBall geodesic generation) */
#ifndef WUBU_PFLOW_H
#define WUBU_PFLOW_H
#ifdef __cplusplus
extern "C" {
#endif
void wubu_pf_velocity(const float* z_data,const float* z_noise,
                      int D,float c,float* vel);
void wubu_pf_step(float* z,const float* z_data,int D,float c,float dt);
void wubu_pf_trajectory(const float* z_noise,const float* z_data,
                        int D,float c,int steps,float* z_out);
#ifdef __cplusplus
}
#endif
#endif
