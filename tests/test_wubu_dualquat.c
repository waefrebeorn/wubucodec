#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_dualquat.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Dual Quaternion Tests ===\n\n");

    printf("  g1_translation_roundtrip...");
    {
        /* create with known translation, extract it back */
        float dq[8];
        wubu_dq_create(0.1f,0,0,1, 3.0f,-2.0f,5.0f, dq);
        float t[3];
        wubu_dq_get_translation(dq,t);
        printf("[t=(%.2f,%.2f,%.2f)] ",t[0],t[1],t[2]);
        CHECK(fabsf(t[0]-3.0f)<0.01f);
        CHECK(fabsf(t[1]+2.0f)<0.01f);
        CHECK(fabsf(t[2]-5.0f)<0.01f);
    }
    printf("PASS\n");passed++;

    printf("  g2_transform_rotation...");
    {
        /* rotate (1,0,0) by 90° around Z: should give (0,1,0) */
        float dq[8];
        wubu_dq_create(M_PI/2,0,0,1, 0,0,0, dq);
        float p[3]={1,0,0},result[3];
        wubu_dq_transform(dq,p,result);
        printf("[r=(%.3f,%.3f)] ",result[0],result[1]);
        CHECK(fabsf(result[0])<0.01f);
        CHECK(fabsf(result[1]-1.0f)<0.01f);
    }
    printf("PASS\n");passed++;

    printf("  g3_transform_translation...");
    {
        /* pure translation of (0,0,0) by (5,3,-1): should give (5,3,-1) */
        float dq[8];
        wubu_dq_create(0,0,0,1, 5,3,-1, dq);
        float p[3]={0,0,0},result[3];
        wubu_dq_transform(dq,p,result);
        printf("[r=(%.2f,%.2f,%.2f)] ",result[0],result[1],result[2]);
        CHECK(fabsf(result[0]-5)<0.01f);
        CHECK(fabsf(result[1]-3)<0.01f);
        CHECK(fabsf(result[2]+1)<0.01f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
