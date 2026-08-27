#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_mvprep.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Skip + Merge + AMVP Tests ===\n\n");
    const int W=64,H=64,BS=8;

    printf("  g1_skip_identical...");
    {
        uint8_t* f1=malloc((size_t)W*H);
        uint8_t* f2=malloc((size_t)W*H);
        memset(f1,128,W*(size_t)H);memset(f2,128,W*(size_t)H);
        CHECK(wubu_skip_detect(f1,f2,W,H,16,16,BS,100)==1);
        free(f1);free(f2);
    }
    printf("PASS\n");passed++;

    printf("  g2_no_skip_different...");
    {
        uint8_t* f1=malloc((size_t)W*H);
        uint8_t* f2=malloc((size_t)W*H);
        memset(f1,50,W*(size_t)H);memset(f2,200,W*(size_t)H);
        CHECK(wubu_skip_detect(f1,f2,W,H,16,16,BS,100)==0);
        free(f1);free(f2);
    }
    printf("PASS\n");passed++;

    printf("  g3_merge_candidates...");
    {
        int bpr=W/BS; /* blocks per row = 8 */
        int n_blocks=bpr*(H/BS);
        int16_t* mv_field=calloc(n_blocks*2,sizeof(int16_t));
        uint8_t* avail=calloc(n_blocks,1);
        
        /* set some neighbor MVs */
        int target=bpr+3; /* block at row=1,col=3 */
        mv_field[(target-1)*2]=-4; mv_field[(target-1)*2+1]=2; /* left */
        avail[target]|=0x01;
        mv_field[(target-bpr)*2]=6; mv_field[(target-bpr)*2+1]=-3; /* above */
        avail[target]|=0x02;

        int16_t cands[5*2];
        int n=wubu_merge_candidates(mv_field,avail,bpr,target,cands);
        printf("[%d candidates] ",n);
        CHECK(n>=2); /* should find at least the 2 available neighbors */
        
        /* verify first candidate matches left MV */
        CHECK(cands[0]==-4&&cands[1]==2);
        
        free(mv_field);free(avail);
    }
    printf("PASS\n");passed++;

    printf("  g4_amvp_roundtrip...");
    {
        int16_t actual[2]={15,-7};
        int16_t cands[2][2]={{10,-5},{20,-10}};
        int best_idx;
        int16_t mvd[2];
        
        wubu_amvp_select(actual,(const int16_t*)cands,2,&best_idx,mvd);
        printf("[MVP=%d MVD=(%d,%d)] ",best_idx,mvd[0],mvd[1]);
        
        /* reconstruct at decoder */
        int16_t recon[2];
        wubu_amvp_reconstruct(cands[best_idx],mvd,recon);
        CHECK(recon[0]==actual[0]&&recon[1]==actual[1]);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
