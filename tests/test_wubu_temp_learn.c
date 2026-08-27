#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_temp_learn.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Learned Temperature Tests ===\n\n");
    const int D=8,NNEG=5;
    float c=1.0f;
    /* anchor close to positive, far from negatives */
    float anc[D],pos[D],neg[NNEG*D];
    for(int d=0;d<D;d++){anc[d]=0.05f*d;pos[d]=anc[d]+0.15f;}
    for(int k=0;k<NNEG;k++)
        for(int d=0;d<D;d++)neg[k*D+d]=anc[d]+0.3f*(k%2?-1:1)+d*0.02f;

    printf("  g1_loss_decreases...");
    {
        float lt=logf(0.07f);   /* CLIP init */
        float first=0,last=0;
        for(int step=0;step<100;step++){
            float l=wubu_tl_infonce(anc,pos,neg,NNEG,D,c,&lt,0.1f);
            if(step==0)first=l;if(step==99)last=l;
        }
        printf("[%.3f->%.3f] ",first,last);
        CHECK(last<first);
    }
    printf("PASS\n");passed++;
    printf("  g2_tau_moves...");
    {
        float lt=logf(0.07f),lt0=lt;
        for(int s=0;s<50;s++)
            wubu_tl_infonce(anc,pos,neg,NNEG,D,c,&lt,0.1f);
        CHECK(lt!=lt0);
    }
    printf("PASS\n");passed++;
    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
