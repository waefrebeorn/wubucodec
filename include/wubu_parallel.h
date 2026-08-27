/* GAP-C077: multi-threaded parallel encoding */
#ifndef WUBU_PARALLEL_H
#define WUBU_PARALLEL_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ThreadPool ThreadPool;
ThreadPool* wubu_tp_create(int num_threads);
void wubu_tp_destroy(ThreadPool* tp);
void wubu_tp_run(ThreadPool* tp,void(**jobs)(void*),void** args,int count);
int  wubu_tp_num_threads(ThreadPool* tp);
void wubu_par_dct_quantize(const uint8_t* residual,int* output,
                            int W,int H,int quality,ThreadPool* tp);
#ifdef __cplusplus
}
#endif
#endif
