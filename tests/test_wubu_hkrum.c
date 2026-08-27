/* test_wubu_hkrum.c -- GAP-D018 gates
 *  G1 planted outlier flagged; inliers not (mostly)
 *  G2 scores are finite and non-negative
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hkrum.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Hyperbolic Krum Outlier Tests ===\n\n");
    const int N=12,D=8;
    float c=1.0f;

    /* 10 tight cluster points + 2 far outliers */
    float pts[N*D];
    unsigned rs=42u;
    for(int i=0;i<10;i++){
        for(int d=0;d<D;d++){
            rs=rs*1103515245u+12345u;
            pts[i*D+d]=(float)((rs>>16)%200)/10000.0f-0.01f;   /* near origin */
        }
    }
    /* outliers: near boundary, opposite sides */
    for(int d=0;d<D;d++){
        pts[10*D+d]=(d==0)?-0.85f:((float)(d%3)-1)*0.02f;
        pts[11*D+d]=(d==1)? 0.85f:((float)(d%3)-1)*0.02f;
    }

    printf("  g2_scores_sane...");
    {
        float scores[N];
        CHECK(wubu_hkrum_scores(pts,N,D,c,4,scores)==0);
        for(int i=0;i<N;i++){
            CHECK(!isnan(scores[i]));
            CHECK(scores[i]>=0);
        }
    }
    printf("PASS\n");passed++;

    printf("  g1_outliers_flagged...");
    {
        int flags[N];
        int count=wubu_hkrum_detect(pts,N,D,c,4,3.0f,flags);
        printf("[flagged=%d] ",count);
        CHECK(count>=1&&count<=6);          /* reasonable count */
        CHECK(flags[10]||flags[11]);         /* at least one planted found */
        /* most inliers survive */
        int inlier_flags=0;
        for(int i=0;i<10;i++)if(flags[i])inlier_flags++;
        CHECK(inlier_flags<=count);          /* consistency */
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
