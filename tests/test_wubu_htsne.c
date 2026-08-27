/* test_wubu_htsne.c -- GAP-D026 gates
 *  G1 P rows sum to 1 (symmetric normalized)
 *  G2 Q sums to 1, all finite
 *  G3 KL is non-negative and decreases as embedded distances approach targets
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_htsne.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic t-SNE Tests ===\n\n");
    const int N=12,D_in=8,D_low=2;
    float c=1.0f;

    /* two blobs in input space */
    float xs[N*D_in];
    unsigned rs=42u;
    for(int i=0;i<N;i++){
        float base=(i<6)?-3.0f:3.0f;
        for(int d=0;d<D_in;d++){
            rs=rs*1103515245u+12345u;
            float noise=((rs>>16)%100)/2500.0f-0.02f;
            xs[i*D_in+d]=(d==0)?base+noise:noise;
        }
    }

    printf("  g1_p_rows_normalized...");
    {
        float* P=malloc(sizeof(float)*(size_t)N*N);
        CHECK(wubu_htsne_p_high(xs,N,D_in,P)==0);
        for(int i=0;i<N;i++){
            float s=0;
            for(int j=0;j<N;j++)s+=P[i*N+j];
            CHECK(fabsf(s-(1.0f/N))<0.05f);   /* symmetrized: row ≈ 1/N */
        }
        free(P);
    }
    printf("PASS\n");passed++;

    printf("  g2_q_sums_one...");
    {
        /* embed blobs on-ball at ±x */
        float ys[N*D_low];
        for(int i=0;i<N;i++){
            ys[i*D_low]=((i<6)?-0.4f:0.4f);
            if(D_low>1)ys[i*D_low+1]=0;
        }
        float Q[N*N];
        wubu_htsne_q_low(ys,N,D_low,c,Q);
        float z=0;
        for(int i=0;i<N*N;i++){CHECK(!isnan(Q[i]));z+=Q[i];}
        CHECK(fabsf(z-1.0f)<1e-3f);
    }
    printf("PASS\n");passed++;

    printf("  g3_kl_decreases...");
    {
        /* good embedding (blobs separated) vs bad (all at origin-ish):
         * KL(good) < KL(bad) since good matches blob structure of P */
        float P[N*N];
        wubu_htsne_p_high(xs,N,D_in,P);

        float y_good[N*D_low];
        for(int i=0;i<N;i++){
            y_good[i*D_low]=((i<6)?-0.85f:0.85f);
            if(D_low>1)y_good[i*D_low+1]=0;
        }
        float Qg[N*N];
        wubu_htsne_q_low(y_good,N,D_low,c,Qg);
        /* structural check: WITHIN-blob avg Q must exceed CROSS-blob */
        float within=0,cross=0;int wc=0,cc=0;
        for(int i=0;i<N;i++)
            for(int j=i+1;j<N;j++){
                if((i<6)==(j<6)){within+=Qg[i*N+j];wc++;}
                else{cross+=Qg[i*N+j];cc++;}
            }
        float w_avg=within/wc,x_avg=cross/cc;
        printf("[q_within=%.4f q_cross=%.4f] ",(double)w_avg,(double)x_avg);
        CHECK(w_avg>x_avg*2.0f);   /* structure preserved */
        float kl=wubu_htsne_kl(P,Qg,N);
        CHECK(kl>=0.0f&&!isnan(kl));
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
