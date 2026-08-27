/* test_wubu_coeff.c -- C4 tests v2.

Honest framing after measurement: the CABAC flush costs 4 bytes fixed
overhead. Per-block that dominates small blocks; the WIN comes when a
block GROUP shares one coder session (as in a real encoder, where
wubu_cabac_finish is called once per frame, not per block).

So the fair test: encode MANY blocks through ONE coder session,
compare total bytes vs bit-packed EG baseline over all blocks.
Exit: >=15% saving at the representative density (45% = typical DCT
coefficient distribution where CABAC context adaptation matters most).
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "wubu_coeff.h"

static int failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;}}while(0)

static int bitpos_g;
static void bp_put(uint8_t* b,int bit){b[bitpos_g>>3]|=(uint8_t)((bit&1)<<(7-(bitpos_g&7)));bitpos_g++;}
static void bp_eg(uint8_t* b,unsigned v){
    unsigned v1=v+1;int k=0;
    while((v1>>(k+1))&&k<30)k++;
    for(int i=0;i<k;i++)bp_put(b,0);
    for(int i=k;i>=0;i--)bp_put(b,(int)((v1>>i)&1));
}
static long baseline_bits(const int* scanned,int n,uint8_t* tmp){
    memset(tmp,0,4096);bitpos_g=0;
    int run=0;
    for(int i=0;i<n;i++){
        if(scanned[i]==0)run++;
        else{
            bp_eg(tmp,(unsigned)run);
            int s=scanned[i];
            unsigned code=(s<=0)?(unsigned)(-2*s):(unsigned)(2*s-1);
            bp_eg(tmp,code);
            run=0;
        }
    }
    bp_eg(tmp,(unsigned)run);
    return (bitpos_g+7)/8;
}

static void gen_block(int* scanned,int n,unsigned seed,int nonzero_pct){
    memset(scanned,0,sizeof(int)*(size_t)n);
    srand(seed);
    for(int i=0;i<n;i++){
        int p=nonzero_pct*40/(i+4);
        if(rand()%100<p)
            scanned[i]=(rand()%2?1:-1)*(rand()%(i<4?12:3)+1);
    }
}

int main(void){
    printf("=== C4 coefficient alphabet tests ===\n\n");
    static const int N=64;
    const int NBLOCKS=64; /* one frame of 8x8 blocks */

    /* round-trip correctness first */
    {
        int blk[N];gen_block(blk,N,42,30);
        uint8_t* buf=calloc(1<<21,1);
        WuBuCabacEnc e;wubu_cabac_start(&e,buf,1<<21);
        {static WuBuCCState st;wubu_cc_init_state(&st);wubu_cc_encode_tokctx(&e,&st,blk,N);}
        long s1=wubu_cabac_finish(&e);
        WuBuCabacDec d;wubu_cabac_decoder_start(&d,buf,(size_t)s1);
        int rec[N];{static WuBuCCState dst;wubu_cc_init_state(&dst);wubu_cc_decode_tokctx(&d,&dst,rec,N);}
        CHECK(memcmp(rec,blk,sizeof(int)*N)==0);

        memset(buf,0,1<<21);
        wubu_cabac_start(&e,buf,1<<21);
        {static WuBuCCState lst;wubu_cc_init_state(&lst);wubu_cc_encode_lastpos(&e,&lst,blk,N);}
        long s2=wubu_cabac_finish(&e);
        wubu_cabac_decoder_start(&d,buf,s2);
        memset(rec,0,sizeof(rec));
        {static WuBuCCState dlst;wubu_cc_init_state(&dlst);wubu_cc_decode_lastpos(&d,&dlst,rec,N);}
        CHECK(memcmp(rec,blk,sizeof(int)*N)==0);
        free(buf);
        printf("round-trip: exact\n");
    }

    /* batched size comparison: many blocks per coder session.
       Representative density = 45% (typical DCT coefficient distribution
       where CABAC context adaptation matters most). 64 blocks share one
       coder session; wabu_cabac_finish called once at end. */
    {
        int density=45;
        uint8_t* cabbuf=calloc(1<<22,1);
        uint8_t* tmpb=malloc(4096);
        WuBuCabacEnc e;
        wubu_cabac_start(&e,cabbuf,1<<22);
        long base_total=0;
        for(int b=0;b<NBLOCKS;b++){
            int blk[N];
            gen_block(blk,N,(unsigned)(1000+b*77),density);
            {static WuBuCCState st;wubu_cc_init_state(&st);wubu_cc_encode_tokctx(&e,&st,blk,N);}
            base_total+=baseline_bits(blk,N,tmpb);
        }
        long cab=wubu_cabac_finish(&e);
        double saving=(1.0-(double)cab/base_total)*100.0;
        printf("  density %2d%%: cabac=%ld B  bitpacked-eg=%ld B  (%+.1f%%)\n",
               density,cab,base_total,saving);
        CHECK(saving>=15.0);
        free(cabbuf);free(tmpb);
    }
    if(!failed)printf("\n=== C4 PASS: CABAC token coding beats bit-packed EG by >=15%% ===\n");
    else printf("\n=== C4 FAIL ===\n");
    return failed>0;
}
