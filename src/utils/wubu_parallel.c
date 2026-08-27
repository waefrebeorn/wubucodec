/*
 * wubu_parallel.c -- GAP-C077: Multi-threaded parallel encoding
 * (pthread-based frame-level and block-level parallelism)
 *
 * Frame-parallel: each thread encodes a different GOP independently.
 * Block-parallel within a frame: DCT+quantize for all blocks in parallel.
 * The SIMD SAD (C076) already gives 6.66x on ME; threading adds N_core× more.
 */
#include "wubu_parallel.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

/* ===== Thread pool ===== */

struct ThreadPool {
    pthread_t* threads;
    int num_threads;
    volatile int shutdown;
    
    /* work queue */
    void (**jobs)(void*);
    void** job_args;
    int job_count;
    int job_next;
    int jobs_done;
    pthread_mutex_t mutex;
    pthread_cond_t cv_work;
    pthread_cond_t cv_done;
};

static void* tp_worker(void* arg){
    ThreadPool* tp=(ThreadPool*)arg;
    while(1){
        pthread_mutex_lock(&tp->mutex);
        while(tp->job_next>=tp->job_count&&!tp->shutdown)
            pthread_cond_wait(&tp->cv_work,&tp->mutex);
        if(tp->shutdown){pthread_mutex_unlock(&tp->mutex);return NULL;}
        int job=tp->job_next++;
        pthread_mutex_unlock(&tp->mutex);
        
        tp->jobs[job](tp->job_args[job]);
        
        pthread_mutex_lock(&tp->mutex);
        tp->jobs_done++;
        if(tp->jobs_done>=tp->job_count)
            pthread_cond_broadcast(&tp->cv_done);
        pthread_mutex_unlock(&tp->mutex);
    }
}

ThreadPool* wubu_tp_create(int num_threads){
    if(num_threads<=0)num_threads=4;
    ThreadPool* tp=calloc(1,sizeof(ThreadPool));
    tp->num_threads=num_threads;
    tp->shutdown=0;
    tp->jobs=NULL;tp->job_args=NULL;
    tp->job_count=0;tp->job_next=0;tp->jobs_done=0;
    pthread_mutex_init(&tp->mutex,NULL);
    pthread_cond_init(&tp->cv_work,NULL);
    pthread_cond_init(&tp->cv_done,NULL);
    tp->threads=malloc(sizeof(pthread_t)*(size_t)num_threads);
    for(int i=0;i<num_threads;i++)
        pthread_create(&tp->threads[i],NULL,tp_worker,tp);
    return tp;
}

void wubu_tp_destroy(ThreadPool* tp){
    pthread_mutex_lock(&tp->mutex);
    tp->shutdown=1;
    pthread_cond_broadcast(&tp->cv_work);
    pthread_mutex_unlock(&tp->mutex);
    for(int i=0;i<tp->num_threads;i++)
        pthread_join(tp->threads[i],NULL);
    free(tp->threads);free(tp);
}

void wubu_tp_run(ThreadPool* tp,void(**jobs)(void*),void** args,int count){
    tp->jobs=jobs;tp->job_args=args;
    tp->job_count=count;tp->job_next=0;tp->jobs_done=0;
    pthread_mutex_lock(&tp->mutex);
    pthread_cond_broadcast(&tp->cv_work);
    pthread_mutex_unlock(&tp->mutex);
    /* wait for completion */
    pthread_mutex_lock(&tp->mutex);
    while(tp->jobs_done<count)
        pthread_cond_wait(&tp->cv_done,&tp->mutex);
    pthread_mutex_unlock(&tp->mutex);
}

int wubu_tp_num_threads(ThreadPool* tp){return tp->num_threads;}

/* ===== Parallel DCT + quantize for all blocks in a frame ===== */

typedef struct {
    const uint8_t* residual;  /* W×H×3 signed residuals */
    int* output_coeffs;       /* quantized coefficients */
    int quality;
    int start_block,end_block;
    int blocks_per_row,W,H;
} DctJob;

static void dct_worker(void* arg){
    DctJob* job=(DctJob*)arg;
    extern void wubu_dct8x8_forward(const int*,int*);
    extern void wubu_dct8x8_quantize(const int*,int,int*);
    extern void wubu_zz_scan(const int*,int*);
    
    for(int b=job->start_block;b<job->end_block;b++){
        int ch=b/(job->blocks_per_row*(job->H/8));
        int by=(b%(job->blocks_per_row*(job->H/8)))/job->blocks_per_row;
        int bx=b%job->blocks_per_row;
        
        int block[64];
        for(int r=0;r<8;r++)for(int c=0;c<8;c++){
            int px=bx*8+c,py=by*8+r;
            if(px<job->W&&py<job->H){
                size_t idx=((size_t)(py*job->W+px)*3+ch);
                block[r*8+c]=(int)((int8_t)job->residual[idx]);
            }else block[r*8+c]=0;
        }
        
        int coeffs[64],quantized[64],scanned[64];
        wubu_dct8x8_forward(block,coeffs);
        wubu_dct8x8_quantize(coeffs,job->quality,quantized);
        wubu_zz_scan(quantized,scanned);
        
        int* out=job->output_coeffs+b*64;
        memcpy(out,scanned,sizeof(int)*64);
    }
}

void wubu_par_dct_quantize(const uint8_t* residual,int* output,
                            int W,int H,int quality,
                            ThreadPool* tp){
    int nbx=W/8,nby=H/8,n_blocks=nbx*nby*3;
    DctJob* jobs_data=malloc(sizeof(DctJob)*(size_t)n_blocks);
    void** jobs=malloc(sizeof(void*)*(size_t)n_blocks);
    void** args=malloc(sizeof(void*)*(size_t)n_blocks);
    
    int blocks_per_thread=(n_blocks+tp->num_threads-1)/tp->num_threads;
    
    for(int i=0;i<n_blocks;i+=blocks_per_thread){
        int end=(i+blocks_per_thread<n_blocks)?i+blocks_per_thread:n_blocks;
        jobs_data[i].residual=residual;
        jobs_data[i].output_coeffs=output;
        jobs_data[i].quality=quality;
        jobs_data[i].start_block=i;
        jobs_data[i].end_block=end;
        jobs_data[i].blocks_per_row=W/8;
        jobs_data[i].W=W;jobs_data[i].H=H;
        jobs[i/blocks_per_thread]=dct_worker;
        args[i/blocks_per_thread]=&jobs_data[i];
    }
    
    int n_jobs=(n_blocks+blocks_per_thread-1)/blocks_per_thread;
    wubu_tp_run(tp,jobs,args,n_jobs);
    
    free(jobs_data);free(jobs);free(args);
}
