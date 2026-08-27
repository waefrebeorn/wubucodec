/* test_wubu_hanalogy.c -- GAP-D035 gates
 *  G1 constructed analogy: vocab with a,b,c,d where d = c + (a-b)
 *     offset exactly → lookup returns d
 *  G2 log0 round trip: exp0(log0(x)) == x
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_hanalogy.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic Analogy Tests ===\n\n");
    const int V=6,D=8;
    float c=1.0f;

    /* construct: a at norm .3 dir e0; b at norm .15 dir e0 (offset = radial out);
     * c along e1 at .2; d along e1 at .35 (= c + offset in tangent space) */
    float emb[V*D];
    memset(emb,0,sizeof(emb));
    emb[0*D+0]=0.30f;                 /* a */
    emb[1*D+0]=0.15f;                 /* b */
    emb[2*D+1]=0.20f;                 /* c */
    emb[3*D+1]=0.35f;                 /* d: the "answer" */
    emb[4*D+2]=0.25f;                 /* distractor */
    emb[5*D+3]=0.10f;                 /* distractor */

    printf("  g1_analogy_finds_d...");
    {
        unsigned seed=42u;
        int ans=wubu_ha_analogy(emb,V,D,c,0,1,2,&seed);
        printf("[ans=%d want=3] ",ans);
        CHECK(ans==3);
    }
    printf("PASS\n");passed++;

    printf("  g2_log_roundtrip...");
    {
        float x[D],lg[D],back[D];
        unsigned rs=5u;
        for(int d=0;d<D;d++){
            rs=rs*1103515245u+12345u;
            x[d]=((float)((rs>>16)%2000))/20000.0f-0.05f;
        }
        wubu_ha_log0(x,D,c,lg);
        /* exp_0(lg): coeff=tanh(sqrt(c)|v|)/(sqrt(c)|v|)*v */
        float n2=0;
        for(int d=0;d<D;d++)n2+=lg[d]*lg[d];
        float nv=sqrtf(n2);
        float coeff=tanhf(sqrtf(c)*nv)/(sqrtf(c)*nv);
        for(int d=0;d<D;d++)back[d]=coeff*lg[d];
        for(int d=0;d<D;d++)CHECK(fabsf(back[d]-x[d])<1e-4f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
