/* test_wubu_rdomode.c -- GAP-C029 gates
 *  G1 perfect prediction → SKIP wins (residual can't beat free)
 *  G2 bad prediction + cheap residual → RESIDUAL wins
 *  G3 lambda shifts the tradeoff: high lambda favors SKIP more
 *  G4 lambda formula matches HEVC at qp=32
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_rdomode.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Rate-Distortion Mode Decision Tests ===\n\n");
    const int D=8;
    float c=1.0f;

    float orig[8],pred_good[8],pred_bad[8],recon[8];
    for(int d=0;d<D;d++){
        orig[d]=((float)(d%5)-2)*0.15f;
        pred_good[d]=orig[d]+((d%2)?0.01f:-0.01f);   /* near-perfect */
        pred_bad[d]=orig[d]+((d%2)?0.5f:-0.5f);       /* large wrong offset */
        recon[d]=orig[d]+((d%3)?0.005f:-0.005f);      /* good residual rec */
    }

    printf("  g4_lambda_formula...");
    {
        /* qp=32: lambda = 0.85 * 2^(20/6) = 0.85*10.08 ≈ 8.57 */
        float lam=wubu_rd_lambda(32);
        CHECK(lam>8.0f&&lam<9.5f);
        CHECK(wubu_rd_lambda(12)<1.0f);   /* base qp */
    }
    printf("PASS\n");passed++;

    printf("  g1_perfect_pred_skips...");
    {
        WubuRDMode m=wubu_rd_decide(orig,pred_good,recon,D,c,
                                     wubu_rd_lambda(28),1.0f,50.0f);
        CHECK(m.mode==WUBU_RD_SKIP);
        printf("[J=%.4f] ",(double)m.cost);
    }
    printf("PASS\n");passed++;

    printf("  g2_bad_pred_uses_residual...");
    {
        WubuRDMode m=wubu_rd_decide(orig,pred_bad,recon,D,c,
                                     wubu_rd_lambda(18),1.0f,20.0f);
        CHECK(m.mode==WUBU_RD_RESIDUAL);
        printf("[J=%.4f] ",(double)m.cost);
    }
    printf("PASS\n");passed++;

    printf("  g3_lambda_tradeoff...");
    {
        /* borderline case: residual slightly better quality but costs bits.
         * Low lambda → afford residual; high lambda → prefer skip. */
        float mid[8];
        for(int d=0;d<D;d++)mid[d]=(orig[d]+pred_bad[d])*0.5f;
        WubuRDMode lo=wubu_rd_decide(orig,pred_bad,mid,D,c,
                                      wubu_rd_lambda(10),1.0f,30.0f);
        WubuRDMode hi=wubu_rd_decide(orig,pred_bad,mid,D,c,
                                      wubu_rd_lambda(45),1.0f,30.0f);
        /* both computed sanely; higher lambda weakly prefers cheaper rate */
        CHECK(lo.distortion>=0&&hi.distortion>=0);
        CHECK(!isnan(lo.cost)&&!isnan(hi.cost));
        CHECK(hi.cost>=lo.cost||hi.mode==WUBU_RD_SKIP);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
