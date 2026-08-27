/* Maximum speed utilization pipeline */
#ifndef WUBU_SPEED_H
#define WUBU_SPEED_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n_threads;
    int use_avx2;
    int use_sse41;
    int gop_parallel;
    int frame_parallel;
    int wavefront;
} WubuSpeedConfig;

double wubu_time_now(void);
void wubu_speed_config(WubuSpeedConfig* cfg,int n_threads);
double wubu_bench_me_throughput(const uint8_t* curr,const uint8_t* ref,
                                  int W,int H,int seconds);
long wubu_parallel_encode(const uint8_t* frames,int n_frames,
                            int W,int H,int gop_size,int quality,
                            int n_threads,long* out_duration_us);

typedef struct { int has_avx2,has_sse41,n_cores; } WubuSpeedProfile;

#ifdef __cplusplus
}
#endif
#endif
