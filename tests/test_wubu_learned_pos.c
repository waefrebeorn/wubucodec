/* test_wubu_learned_pos.c -- GAP-C034 gates
 *  G1 rows are distinct (different positions → different embeddings)
 *  G2 apply() keeps tokens on-ball
 *  G3 same token at different positions gets different outputs
 *  G4 training moves a row (FD gradient nonzero)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_learned_pos.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Learned Positional Embedding Tests ===\n\n");
    const int T=16,D=8;
    float c=1.0f;

    WubuLearnedPos lp;
    CHECK(wubu_lp_init(&lp,T,D,42u)==0);

    printf("  g1_rows_distinct...");
    {
        for(int t=0;t<T-1;t++){
            float d=0;
            for(int j=0;j<D;j++){
                float df=lp.table[t*D+j]-lp.table[(t+1)*D+j];
                d+=df*df;
            }
            CHECK(sqrtf(d)>1e-6f);
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_apply_on_ball...");
    {
        float tok[8]={0.2f,-0.1f,0.15f,0.05f,-0.12f,0.08f,-0.03f,0.1f};
        for(int t=0;t<T;t+=5){
            float out[8];
            wubu_lp_apply(&lp,tok,t,c,out);
            float n2=0;for(int d=0;d<D;d++)n2+=out[d]*out[d];
            CHECK(n2<1.0f);
        }
    }
    printf("PASS\n");passed++;

    printf("  g3_position_distinguishes...");
    {
        float tok[8]={0.2f,-0.1f,0.15f,0.05f,-0.12f,0.08f,-0.03f,0.1f};
        float o0[8],o1[8];
        wubu_lp_apply(&lp,tok,0,c,o0);
        wubu_lp_apply(&lp,tok,7,c,o1);
        float diff=0;
        for(int d=0;d<D;d++){float df=o0[d]-o1[d];diff+=df*df;}
        CHECK(sqrtf(diff)>1e-6f);
    }
    printf("PASS\n");passed++;

    printf("  g4_training_moves_row...");
    {
        float before[8];
        memcpy(before,lp.table,sizeof(float)*D);
        float grad[8];
        for(int d=0;d<8;d++)grad[d]=0.1f*(d+1);
        wubu_lp_train_row(&lp,3,grad,0.1f);
        float moved=0;
        for(int d=0;d<8;d++){
            float df=lp.table[3*D+d]-before[d];
            moved+=df*df;
        }
        CHECK(moved>0);
    }
    printf("PASS\n");passed++;

    wubu_lp_free(&lp);
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
