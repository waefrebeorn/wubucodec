/* wubu_thread_pool.h — persistent thread pool for frame-level parallelism */
#ifndef WUBU_THREAD_POOL_H
#define WUBU_THREAD_POOL_H
#include <stdint.h>

#define WUBU_MAX_THREADS 8

/* Initialize thread pool with N worker threads */
void wubu_thread_pool_init(int num_threads);

/* Shutdown thread pool */
void wubu_thread_pool_shutdown(void);

/* Encode frame using thread pool (horizontal strip decomposition) */
int wubu_encode_frame_pooled(const uint8_t* orig, const uint8_t* ref,
                              int W, int H, int qp, int frame_type,
                              uint8_t* recon, long* total_bits, int preset);

#endif
