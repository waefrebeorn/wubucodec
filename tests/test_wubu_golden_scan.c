#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_golden_scan.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Golden Scan Tests ===\n\n");
    printf("  g1_coverage_uniformity...");
    {
        /* golden-angle samples should fill the disk uniformly:
         * split into quadrants, all should get ~25% of points */
        int n=1000;
        float x[n],y[n];
        wubu_gs_order(n,x,y);
        int q[4]={0};
        for(int i=0;i<n;i++){
            int qx=x[i]>0,qy=y[i]>0;
            q[qx*2+qy]++;
        }
        for(int i=0;i<4;i++)CHECK(q[i]>n/4*0.7&&q[i]<n/4*1.3);
    }
    printf("PASS\n");passed++;
    printf("  g2_progressive_coverage...");
    {
        /* THE golden-angle property: any prefix of N samples covers ALL
         * angular sectors uniformly (progressive acquisition). Raster
         * scan covers only a strip at 10% progress. */
        int n=500;
        float gx[500],gy[500];
        wubu_gs_order(n,gx,gy);
        /* at 20% prefix, count how many of 8 angular octants have points */
        int prefix=n/5;
        int octants[8]={0};
        for(int i=0;i<prefix;i++){
            float ang=atan2f(gy[i],gx[i]);
            if(ang<0)ang+=6.28318f;
            int oct=(int)(ang/(6.28318f/8));
            if(oct>=8)oct=7;
            octants[oct]++;
        }
        int filled=0;
        for(int i=0;i<8;i++)if(octants[i]>0)filled++;
        printf("[octants %d/8 at 20%%] ",filled);
        CHECK(filled==8);   /* ALL octants covered at just 20% progress */

        /* raster baseline at same fraction: covers only ~1 row band */
        int side=(int)sqrtf((float)n);
        float rx[1024],ry[1024];
        wubu_gs_raster(side,rx,ry);
        int r_octants[8]={0};
        for(int i=0;i<prefix;i++){
            float ang=atan2f(ry[i],rx[i]);
            if(ang<0)ang+=6.28318f;
            int oct=(int)(ang/(6.28318f/8));
            if(oct>=8)oct=7;
            r_octants[oct]++;
        }
        int r_filled=0;
        for(int i=0;i<8;i++)if(r_octants[i]>0)r_filled++;
        printf("[raster %d/8] ",r_filled);
        CHECK(r_filled<filled);   /* golden strictly beats raster */
    }
    printf("PASS\n");passed++;
    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
