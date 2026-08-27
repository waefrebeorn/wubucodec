/* test_wubu_kvcache.c -- GAP-C036 gates
 *  G1 append grows len; overflow rejected
 *  G2 decode output on-ball and finite
 *  G3 nearest cached key dominates attention (query = cached key)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_kvcache.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== KV Cache Tests ===\n\n");
    const int MAXT=8,D=8;
    float c=1.0f;

    WubuKVCache kc;
    CHECK(wubu_kvc_init(&kc,MAXT,D,c)==0);

    /* fill cache with 4 distinct on-ball entries */
    float entries[4][8];
    unsigned rs=42u;
    for(int i=0;i<4;i++){
        for(int d=0;d<D;d++){
            rs=rs*1103515245u+12345u;
            entries[i][d]=(float)((rs>>16)%2000)/20000.0f-0.05f;
        }
        float n2=0;for(int d=0;d<D;d++)n2+=entries[i][d]*entries[i][d];
        if(n2>0.8f){float s=sqrtf(0.8f/n2);for(int d=0;d<D;d++)entries[i][d]*=s;}
        CHECK(wubu_kvc_append(&kc,entries[i],entries[i])==0);
    }

    printf("  g1_len_and_overflow...");
    {
        CHECK(kc.len==4);
        CHECK(kc.len==4);
        CHECK(wubu_kvc_append(&kc,entries[0],entries[0])==0);   /* 5 */
        CHECK(wubu_kvc_append(&kc,entries[0],entries[0])==0);   /* 6 */
        CHECK(wubu_kvc_append(&kc,entries[0],entries[0])==0);   /* 7 */
        CHECK(wubu_kvc_append(&kc,entries[0],entries[0])==0);   /* 8 = full */
        CHECK(wubu_kvc_append(&kc,entries[0],entries[0])==-1);  /* overflow */
    }
    printf("PASS\n");passed++;
    wubu_kvc_reset(&kc);

    /* refill with 4 */
    for(int i=0;i<4;i++)wubu_kvc_append(&kc,entries[i],entries[i]);

    printf("  g2_decode_on_ball...");
    {
        float out[8];
        wubu_kvc_decode(&kc,entries[0],0.5f,out);
        float n2=0;for(int d=0;d<D;d++)n2+=out[d]*out[d];
        CHECK(n2<1.0f);
        for(int d=0;d<D;d++)CHECK(!isnan(out[d]));
    }
    printf("PASS\n");passed++;

    printf("  g3_nearest_dominates...");
    {
        /* query exactly = entry[2] → output should be very close to entry[2] */
        float out[8];
        wubu_kvc_decode(&kc,entries[2],0.1f,out);   /* sharp tau */
        float dist=0;
        for(int d=0;d<D;d++){float df=out[d]-entries[2][d];dist+=df*df;}
        CHECK(sqrtf(dist)<0.2f);
    }
    printf("PASS\n");passed++;

    wubu_kvc_free(&kc);
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
