#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <math.h>
#include "wubu_quat_rate.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
int main(void){
    printf("=== Quaternion Rate Control Tests ===\n\n");
    const int NF=20,D=4;

    printf("  g1_velocity_measures_rotation...\n");
    {
        /* identity → identity = zero velocity */
        float qi[4]={1,0,0,0};
        float v0=wubu_qr_angular_velocity(qi,qi);
        CHECK(v0<0.01f);
        /* 90° rotation around Z */
        float q90[4]={cosf(M_PI/4),0,0,sinf(M_PI/4)};
        float v90=wubu_qr_angular_velocity(qi,q90);
        printf("[v=%.3f want %.3f] ",v90,M_PI/2);
        CHECK(fabsf(v90-M_PI/2)<0.01f);
    }
    printf("PASS\n");passed++;

    printf("  g2_fast_gets_more_bits...\n");
    {
        /* slow frame then fast frame */
        float angles[2]={0.01f,0.5f};
        int bits[2];
        wubu_qr_allocate(angles,2,8,4,12,bits);
        CHECK(bits[1]>bits[0]);
    }
    printf("PASS\n");passed++;

    printf("  g3_encode_decode_roundtrip...\n");
    {
        const char* path="/tmp/qr_test.bin";
        FILE* wf=fopen(path,"wb");
        CHECK(wf!=NULL);
        float angle=0.5f,ax=0,ay=0,az=1;
        wubu_qr_encode_delta(angle,ax,ay,az,12,wf);
        fclose(wf);

        FILE* rf=fopen(path,"rb");
        float dax,day,daz;
        float dangle=wubu_qr_decode_delta(rf,12,&dax,&day,&daz);
        fclose(rf);
        printf("[%+.3f vs %+.3f] ",dangle,angle);
        CHECK(fabsf(dangle-angle)<0.001f);
        CHECK(fabsf(daz-az)<0.01f);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
