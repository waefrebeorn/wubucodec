#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#define M_PI 3.14159265358979f
#include "wubu_speed.h"
#include "wubu_parallel.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

double wubu_time_now(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec+ts.tv_nsec/1e9;
}

typedef struct { int has_avx2,has_sse41,n_cores; } SpeedProfileInternal;

static void detect_cpu(SpeedProfileInternal* sp){
    memset(sp,0,sizeof(*sp));
    sp->n_cores=(int)sysconf(_SC_NPROCESSORS_ONLN);
    if(sp->n_cores<1)sp->n_cores=1;
#if defined(__AVX2__)
    sp->has_avx2=1;
#endif
#if defined(__SSE4_1__)
    sp->has_sse41=1;
#endif
}

void wubu_speed_config(WubuSpeedConfig* cfg,int n_threads){
    SpeedProfileInternal sp;
    detect_cpu(&sp);
    cfg->n_threads=n_threads>0?n_threads:sp.n_cores;
    cfg->use_avx2=sp.has_avx2;
    cfg->use_sse41=sp.has_sse41;
    cfg->gop_parallel=cfg->n_threads>=8;
    cfg->frame_parallel=cfg->n_threads>=4;
    cfg->wavefront=cfg->n_threads>=2;
}

extern long wubu_simd_sad(const uint8_t*,const uint8_t*,int);

double wubu_bench_me_throughput(const uint8_t* curr,const uint8_t* ref,
                                  int W,int H,int seconds){
    double start=wubu_time_now();
    long iterations=0;
    while(wubu_time_now()-start<seconds){
        for(int by=0;by<H/16;by++)
            for(int bx=0;bx<W/16;bx++){
                long sad=wubu_simd_sad(curr+(size_t)(by*16)*W+bx*16,
                                        ref+(size_t)(by*16)*W+bx*16,256);
                if(sad==0)break;
            }
        iterations++;
    }
    double elapsed=wubu_time_now()-start;
    return iterations/elapsed;
}
