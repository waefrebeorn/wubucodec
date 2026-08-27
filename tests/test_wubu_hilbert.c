#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_hilbert.h"
#include "wubu_golden_scan.h"

static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)

int main(void){
    printf("=== Hilbert Scan Tests ===\n\n");
    const int SIDE=32,N=SIDE*SIDE;
    float hx[N],hy[N],gx[N],gy[N];

    wubu_hil_order(SIDE,hx,hy);
    wubu_gs_order(N,gx,gy);

    printf("  g1_hilbert_locality...");
    {
        float lh=wubu_gs_locality(hx,hy,N);
        float lg=wubu_gs_locality(gx,gy,N);
        printf("[hilbert=%.4f golden=%.4f] ",lh,lg);
        /* Hilbert should have MUCH better locality (adjacent steps) */
        CHECK(lh<lg);
        CHECK(lh<0.05f);  /* adjacent pixels = tiny step */
    }
    printf("PASS\n");passed++;

    printf("  g2_coverage_complete...");
    {
        /* all positions visited exactly once */
        int visited[SIDE][SIDE];
        memset(visited,0,sizeof(visited));
        for(int i=0;i<N;i++){
            int x=(int)((hx[i]+0.5f)*SIDE);
            int y=(int)((hy[i]+0.5f)*SIDE);
            if(x>=0&&x<SIDE&&y>=0&&y<SIDE)visited[y][x]++;
        }
        for(int y=0;y<SIDE;y++)
            for(int x=0;x<SIDE;x++)
                CHECK(visited[y][x]==1);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
