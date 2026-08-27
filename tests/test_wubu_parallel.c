#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_parallel.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Parallel Encoding Tests ===\n\n");

    printf("  g1_thread_pool_basic...");
    {
        ThreadPool* tp=wubu_tp_create(4);
        CHECK(wubu_tp_num_threads(tp)==4);
        wubu_tp_destroy(tp);
    }
    printf("PASS\n");passed++;

    printf("  g2_parallel_dct_correct...");
    {
        /* create test residual data */
        int W=64,H=64;
        uint8_t* residual=malloc((size_t)W*H*3);
        for(long i=0;i<(long)W*H*3;i++)residual[i]=(uint8_t)((i*7)%256);

        int n_coeffs=W/8*H/8*3*64;
        int* output=malloc(sizeof(int)*(size_t)n_coeffs);

        ThreadPool* tp=wubu_tp_create(4);
        wubu_par_dct_quantize(residual,output,W,H,5,tp);
        wubu_tp_destroy(tp);

        /* verify output is non-zero (transform happened) */
        int nonzero=0;
        for(int i=0;i<n_coeffs;i++)if(output[i]!=0)nonzero++;
        printf("[%d/%d non-zero] ",nonzero,n_coeffs);
        CHECK(nonzero>n_coeffs/10);
        
        free(residual);free(output);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
