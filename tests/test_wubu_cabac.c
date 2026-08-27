#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_cabac.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== CABAC Tests ===\n\n");

    printf("  g1_skew_compression...");
    {
        uint8_t* buf=calloc(4096,1);
        WuBuCabacEnc e;
        wubu_cabac_start(&e,buf,4096);
        
        CabacContext ctx={0,0};
        srand(42);
        int bins[200];
        for(int i=0;i<200;i++){
            bins[i]=rand()%100<95?0:1;
            wubu_cabac_encode_bin(&e,&ctx,bins[i]);
        }
        long size=wubu_cabac_finish(&e);
        float bits_per_bin=(float)size*8/200;
        printf("[%ld bytes = %.2f bits/bin] ",size,bits_per_bin);
        CHECK(size>0&&size<30);
        free(buf);
    }
    printf("PASS\n");passed++;

    printf("  g2_uniform_bypass...");
    {
        uint8_t* buf=calloc(4096,1);
        WuBuCabacEnc e;
        wubu_cabac_start(&e,buf,4096);
        
        srand(42);
        for(int i=0;i<100;i++)
            wubu_cabac_encode_bypass(&e,rand()%2);
        long size=wubu_cabac_finish(&e);
        float bits_bin=(float)size*8/100;
        printf("[%ld bytes = %.1f bits/bin] ",size,bits_bin);
        CHECK(bits_bin>=1.0f&&bits_bin<=2.5f); /* uniform → ~1 bit each */
        free(buf);
    }
    printf("PASS\n");passed++;

    /* ===== C1: decoder round-trip tests ===== */
    printf("  c1_roundtrip_skew...");
    {
        uint8_t* buf=calloc(8192,1);
        WuBuCabacEnc e;
        wubu_cabac_start(&e,buf,8192);
        CabacContext ctx={0,0};
        srand(7);
        static int bins[10000];
        for(int i=0;i<10000;i++){
            bins[i]=rand()%100<90?0:1;
            wubu_cabac_encode_bin(&e,&ctx,bins[i]);
        }
        long size=wubu_cabac_finish(&e);
        WuBuCabacDec d;
        wubu_cabac_decoder_start(&d,buf,(size_t)size);
        /* fresh context with identical init */
        CabacContext dctx={0,0};
        int bad=0;
        for(int i=0;i<10000;i++){
            int b=wubu_cabac_decode_bin(&d,&dctx);
            if(b!=bins[i]){bad=i+1;break;}
        }
        printf("[%.2f bits/bin, first mismatch @ %s] ",size*8.0/10000,bad?"":"none");
        CHECK(bad==0);
        free(buf);
    }
    printf("PASS\n");passed++;

    printf("  c1_roundtrip_bypass...");
    {
        uint8_t* buf=calloc(8192,1);
        WuBuCabacEnc e;
        wubu_cabac_start(&e,buf,8192);
        srand(99);
        static int bins[5000];
        for(int i=0;i<5000;i++){
            bins[i]=rand()%2;
            wubu_cabac_encode_bypass(&e,bins[i]);
        }
        long size=wubu_cabac_finish(&e);
        WuBuCabacDec d;
        wubu_cabac_decoder_start(&d,buf,(size_t)size);
        int bad=0;
        for(int i=0;i<5000;i++){
            int b=wubu_cabac_decode_bypass(&d);
            if(b!=bins[i]){bad=i+1;break;}
        }
        printf("[%ld bytes = %.1f bits/bin, mismatch @ %s] ",size,size*8.0/5000,bad?"X":"none");
        CHECK(bad==0);
        free(buf);
    }
    printf("PASS\n");passed++;

    printf("  c1_roundtrip_mixed_stress...");
    {
        /* 20 contexts, mixed bin/bypass, random patterns incl. all-ones/all-zeros runs */
        uint8_t* buf=calloc(1<<20,1);
        WuBuCabacEnc e;
        wubu_cabac_start(&e,buf,1<<20);
        srand(1234);
        enum{N=64};
        CabacContext ctxs[N];memset(ctxs,0,sizeof(ctxs));
        static int kind[20000],val[20000],ci[20000];
        for(int i=0;i<20000;i++){
            ci[i]=rand()%N;
            kind[i]=rand()%3; /* 0,1=contexted bin; 2=bypass */
            if(kind[i]<2){
                /* biased per-context: even contexts skew 0, odd skew 1 */
                val[i]= (ci[i]%2) ? (rand()%100<80) : (rand()%100>=80);
                wubu_cabac_encode_bin(&e,&ctxs[ci[i]],val[i]);
            }else{
                val[i]=rand()%2;
                wubu_cabac_encode_bypass(&e,val[i]);
            }
        }
        long size=wubu_cabac_finish(&e);
        WuBuCabacDec d;
        wubu_cabac_decoder_start(&d,buf,(size_t)size);
        CabacContext dctxs[N];memset(dctxs,0,sizeof(dctxs));
        int bad=0;
        for(int i=0;i<20000&&!bad;i++){
            int b;
            if(kind[i]<2)b=wubu_cabac_decode_bin(&d,&dctxs[ci[i]]);
            else b=wubu_cabac_decode_bypass(&d);
            if(b!=val[i])bad=i+1;
        }
        printf("[%ld bytes = %.2f bits/bin, mismatch @ %s] ",size,size*8.0/20000,bad?"X":"none");
        CHECK(bad==0);
        free(buf);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
