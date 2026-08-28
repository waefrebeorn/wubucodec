/* wubu_thread_pool.c — persistent thread pool for frame-level parallelism
 *
 * Each thread encodes entire frames independently.
 * The pool persists across frames to avoid thread creation overhead.
 */

#include "wubu_thread_pool.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

typedef struct {
    const uint8_t* orig;
    const uint8_t* ref;
    uint8_t* recon;
    int W, H, qp;
    int preset;
    int frame_type;
    long* total_bits;
} ThreadJob;

static ThreadJob g_jobs[WUBU_MAX_THREADS];
static pthread_t g_threads[WUBU_MAX_THREADS];
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_work_cond = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_done_cond = PTHREAD_COND_INITIALIZER;
static int g_num_threads = 0;
static int g_running = 0;
static int g_work_ready = 0;
static int g_threads_done = 0;

static void* worker_thread(void* arg) {
    int tid = *(int*)arg;
    free(arg);

    while (1) {
        pthread_mutex_lock(&g_mutex);
        while (!g_work_ready && g_running) {
            pthread_cond_wait(&g_work_cond, &g_mutex);
        }
        if (!g_running) {
            pthread_mutex_unlock(&g_mutex);
            break;
        }
        pthread_mutex_unlock(&g_mutex);

        ThreadJob* job = &g_jobs[tid];
        long bits = 0;
        wubu_encode_frame_preset(job->orig, job->ref, NULL,
                                  job->W, job->H, job->qp,
                                  job->frame_type, NULL,
                                  job->recon, &bits, job->preset);
        *job->total_bits = bits;

        pthread_mutex_lock(&g_mutex);
        g_threads_done++;
        if (g_threads_done == g_num_threads) {
            g_work_ready = 0;
            pthread_cond_signal(&g_done_cond);
        }
        pthread_mutex_unlock(&g_mutex);
    }
    return NULL;
}

void wubu_thread_pool_init(int num_threads) {
    if (num_threads > WUBU_MAX_THREADS) num_threads = WUBU_MAX_THREADS;
    g_num_threads = num_threads;
    g_running = 1;
    g_work_ready = 0;
    g_threads_done = 0;

    for (int t = 0; t < num_threads; t++) {
        int* tid = malloc(sizeof(int));
        *tid = t;
        pthread_create(&g_threads[t], NULL, worker_thread, tid);
    }
}

void wubu_thread_pool_shutdown(void) {
    pthread_mutex_lock(&g_mutex);
    g_running = 0;
    g_work_ready = 0;
    pthread_cond_broadcast(&g_work_cond);
    pthread_mutex_unlock(&g_mutex);

    for (int t = 0; t < g_num_threads; t++) {
        pthread_join(g_threads[t], NULL);
    }
}

/* Encode a single frame using thread pool (each thread handles one tile) */
int wubu_encode_frame_pooled(const uint8_t* orig, const uint8_t* ref,
                              int W, int H, int qp, int frame_type,
                              uint8_t* recon, long* total_bits, int preset) {
    if (g_num_threads <= 1) {
        return wubu_encode_frame_preset(orig, ref, NULL, W, H, qp, frame_type, NULL, recon, total_bits, preset);
    }

    /* For now, just use single-threaded — the encoder isn't thread-safe for tiling */
    /* TODO: implement proper tile-based parallelism with thread-safe encoder */
    return wubu_encode_frame_preset(orig, ref, NULL, W, H, qp, frame_type, NULL, recon, total_bits, preset);
}
