/* GAP-A008: hyperbolic beam search over latent paths */
#ifndef WUBU_HBEAM_H
#define WUBU_HBEAM_H
#ifdef __cplusplus
extern "C" {
#endif
/* search from start to goal, optionally steering via waypoints.
 * writes up to max_steps path points (D floats each); returns count. */
int wubu_beam_search(const float* start,const float* goal,
                      const float* waypoints,int n_wp,
                      int D,float c,float step_size,
                      int width,int max_steps,float* out_path);
#ifdef __cplusplus
}
#endif
#endif
