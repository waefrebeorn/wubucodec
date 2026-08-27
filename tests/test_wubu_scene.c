#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_scene.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Scene Analysis Tests ===\n\n");
    const int W=64,H=64;

    printf("  g1_scene_change_detected...");
    {
        uint8_t* f1=malloc((size_t)W*H);
        uint8_t* f2=malloc((size_t)W*H);
        memset(f1,50,(size_t)W*H);   /* dark scene */
        memset(f2,200,(size_t)W*H);  /* bright scene — total change */
        CHECK(wubu_scene_change(f1,f2,W,H,0.3)==1);
        free(f1);free(f2);
    }
    printf("PASS\n");passed++;

    printf("  g2_no_scene_change...");
    {
        uint8_t* f1=malloc((size_t)W*H);
        uint8_t* f2=malloc((size_t)W*H);
        for(long i=0;i<(long)W*H;i++){
            f1[i]=(uint8_t)(i%256);
            f2[i]=(uint8_t)(i%256+5); /* slight change */
        }
        CHECK(wubu_scene_change(f1,f2,W,H,0.3)==0);
        free(f1);free(f2);
    }
    printf("PASS\n");passed++;

    printf("  g3_spatial_complexity...");
    {
        /* flat image → zero complexity */
        uint8_t* flat=calloc((size_t)W*H,1);
        double c_flat=wubu_spatial_complexity(flat,W,H);
        
        /* noise → high complexity */
        uint8_t* noisy=malloc((size_t)W*H);
        srand(42);
        for(long i=0;i<(long)W*H;i++)noisy[i]=rand()%256;
        double c_noise=wubu_spatial_complexity(noisy,W,H);
        
        printf("[flat=%.2f noise=%.2f] ",c_flat,c_noise);
        CHECK(c_noise>c_flat*10);
        free(flat);free(noisy);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
