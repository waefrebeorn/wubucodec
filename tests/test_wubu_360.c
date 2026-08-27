#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_360.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== 360 Video Tests ===\n\n");

    printf("  g1_viewport_fraction...");
    {
        /* 100°×90° FOV at equator ≈ 19.6% of sphere */
        double frac=wubu_viewport_fraction(100,90,0);
        printf("[%.1f%%] ",frac*100);
        CHECK(frac>0.17&&frac<0.22);
    }
    printf("PASS\n");passed++;

    printf("  g2_wraparound_adjust...");
    {
        /* dx=+900 on 176-wide frame → should wrap to -88+... */
        int adjusted=wubu_me_wraparound_adjust(170,176);
        CHECK(adjusted==-6); /* wraps around */
        
        adjusted=wubu_me_wraparound_adjust(-170,176);
        CHECK(adjusted==6);
        
        adjusted=wubu_me_wraparound_adjust(5,176);
        CHECK(adjusted==5); /* no wrap needed */
    }
    printf("PASS\n");passed++;

    printf("  g3_cmp_face_assignment...");
    {
        int fx,fy;
        int face=wubu_cmp_face(0,0,1,&fx,&fy,32); /* +Z = front */
        CHECK(face==1);
        
        face=wubu_cmp_face(1,0,0,&fx,&fy,32); /* +X = right */
        CHECK(face==2);
        
        face=wubu_cmp_face(0,1,0,&fx,&fy,32); /* +Y = top */
        CHECK(face==4);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
