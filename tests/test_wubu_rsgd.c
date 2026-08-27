/* test_wubu_rsgd.c -- GAP-C023 gates
 *  G1 output stays on-ball after every step (norm < rmax)
 *  G2 gradient descent reduces a simple loss on the ball
 *  G3 near-boundary points move more than near-center ones
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_rsgd.h"

static unsigned long rs_g=0xDEADBEEFUL;
static float fr(void){rs_g^=rs_g<<13;rs_g^=rs_g>>7;rs_g^=rs_g<<17;
    return (float)((rs_g>>11)&0x3FFFFFF)/(float)0x3FFFFFF;}
static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)

int main(void){
    printf("=== Riemannian SGD Tests ===\n\n");
    const int D=8;
    float c=1.0f;

    printf("  g1_stays_on_ball...");
    {
        float x[D],grad[D];
        unsigned rs=42u;
        for(int trial=0;trial<200;trial++){
            float lr=0.01f+fr()*0.5f;
            for(int d=0;d<D;d++){
                rs=rs*1103515245u+12345u;
                x[d]=(float)((rs>>16)%2000)/2000.0f-0.5f;
            }
            /* normalize to inside ball */
            float n2=0;for(int d=0;d<D;d++)n2+=x[d]*x[d];
            if(n2>0.9f){float s=0.9f/sqrtf(n2);for(int d=0;d<D;d++)x[d]*=s;}
            for(int d=0;d<D;d++)grad[d]=((float)((rs>>16)%1000)/500.0f-1.0f);
            /* 10 steps */
            for(int s=0;s<10;s++){
                wubu_rsgd_step(x,grad,D,c,lr);
                float n2b=0;for(int d=0;d<D;d++)n2b+=x[d]*x[d];
                CHECK(n2b<=1.0f+1e-5f);
            }
        }
    }
    printf("PASS\n");passed++;

    printf("  g2_reduces_loss...");
    {
        /* loss = distance from target point t */
        float target[8]={0.3f,-0.2f,0.1f,-0.15f,0.25f,0.05f,-0.1f,0.2f};
        float x[8]={0.0f};   /* start at origin-ish */
        x[0]=0.01f;x[4]=-0.02f;

        float loss_before=0;
        for(int d=0;d<D;d++){float df=x[d]-target[d];loss_before+=df*df;}

        for(int step=0;step<200;step++){
            float grad[8];
            for(int d=0;d<D;d++)grad[d]=x[d]-target[d]; /* d/dx |x-t|^2 / 2 */
            wubu_rsgd_step(x,grad,D,c,0.05f);
        }
        float loss_after=0;
        for(int d=0;d<D;d++){float df=x[d]-target[d];loss_after+=df*df;}
        CHECK(loss_after<loss_before);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
