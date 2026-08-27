/* test_wubu_hvae_kl.c -- GAP-D028 gates
 *  G1 KL(q||q) = 0 (same distribution)
 *  G2 KL(q||p) > 0 when distributions differ
 *  G3 KL grows as sigma_q shrinks (sharper q → farther from prior)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hvae_kl.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic VAE KL Tests ===\n\n");
    const int D=8;
    float c=1.0f;

    float mu_q[8],sig_q[8],mu_p[8],sig_p[8];
    for(int d=0;d<D;d++){
        mu_q[d]=((float)(d%5)-2)*0.05f;sig_q[d]=0.15f;
        mu_p[d]=0;sig_p[d]=0.3f;
    }

    printf("  g1_kl_self_zero...");
    {
        float kl=wubu_hvae_kl_estimate(mu_q,sig_q,mu_q,sig_q,D,c,200);
        printf("[%.4f] ",(double)kl);
        CHECK(fabsf(kl)<0.3f);   /* MC noise tolerance */
    }
    printf("PASS\n");passed++;

    printf("  g2_kl_different_positive...");
    {
        /* shift q's mean away from p */
        float shifted[8];
        for(int d=0;d<D;d++)shifted[d]=mu_q[d]+0.3f;
        float kl=wubu_hvae_kl_estimate(shifted,sig_q,mu_p,sig_p,D,c,200);
        CHECK(kl>0.1f);
    }
    printf("PASS\n");passed++;

    printf("  g3_sharper_q_bigger_kl...");
    {
        float sig_wide[8],sig_narrow[8];
        for(int d=0;d<D;d++){sig_wide[d]=0.4f;sig_narrow[d]=0.08f;}
        float kl_wide=wubu_hvae_kl_estimate(mu_q,sig_wide,mu_p,sig_p,D,c,200);
        float kl_narrow=wubu_hvae_kl_estimate(mu_q,sig_narrow,mu_p,sig_p,D,c,300);
        CHECK(kl_narrow>kl_wide);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
