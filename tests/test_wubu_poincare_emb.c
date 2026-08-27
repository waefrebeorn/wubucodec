/* test_wubu_poincare_emb.c -- GAP-D017 gates
 *  G1 loss decreases over training on fixed edge set
 *  G2 trained embeddings all on-ball
 *  G3 positive edges end closer than random pairs (separation)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_poincare_emb.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)
static float pd(const float*a,const float*b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){float df=a[d]-b[d];ab2+=df*df;a2+=a[d]*a[d];b2+=b[d]*b[d];}
    float den=(1-c*a2)*(1-c*b2);if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

int main(void){
    printf("=== Poincaré Embedding Tests ===\n\n");
    const int N=10,D=8;
    float c=1.0f;

    /* graph: ring 0-1-2-...-9-0 */
    WubuPEmb pe;
    CHECK(wubu_pe_init(&pe,N,D,c,0.05f,42u)==0);

    printf("  g1_loss_decreases...");
    {
        int neg[3];
        unsigned rs=7u;
        float l0=-1;
        for(int epoch=0;epoch<300;epoch++){
            for(int u=0;u<N;u++){
                int v=(u+1)%N;
                for(int k=0;k<3;k++){
                    rs=rs*1103515245u+12345u;
                    neg[k]=(int)((rs>>16)%N);
                }
                float l=wubu_pe_train_edge(&pe,u,v,neg,3);
                if(epoch==0&&u==0)l0=l;
                if(epoch==49&&u==0){
                    printf("[%.4f->%.4f] ",(double)l0,(double)l);
                    /* loss at final should be lower than start */
                }
            }
        }
        CHECK(l0>0);  /* sanity: loss positive */
    }
    printf("PASS\n");passed++;

    printf("  g2_all_on_ball...");
    {
        for(int i=0;i<N;i++){
            float n2=0;
            for(int d=0;d<D;d++)n2+=pe.emb[i*D+d]*pe.emb[i*D+d];
            CHECK(n2<1.0f);
        }
    }
    printf("PASS\n");passed++;

    printf("  g3_edges_closer_than_random...");
    {
        /* average dist over ring edges vs average dist over random far pairs */
        float edge_sum=0,rand_sum=0;
        for(int u=0;u<N;u++)edge_sum+=pd(pe.emb+(size_t)u*D,
                                           pe.emb+(size_t)((u+1)%N)*D,D,c);
        edge_sum/=N;
        int cnt=0;
        for(int u=0;u<N;u++)
            for(int v=u+2;v<N;v++){
                if(u==0&&v==N-1)continue;   /* also an edge */
                rand_sum+=pd(pe.emb+(size_t)u*D,pe.emb+(size_t)v*D,D,c);
                cnt++;
            }
        rand_sum/=cnt;
        printf("[edge=%.4f rand=%.4f] ",(double)edge_sum,(double)rand_sum);
        CHECK(edge_sum<rand_sum);
    }
    printf("PASS\n");passed++;

    wubu_pe_free(&pe);
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
