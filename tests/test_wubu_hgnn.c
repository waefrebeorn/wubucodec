#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979f
#endif
#include "wubu_hgnn.h"
static int passed=0,failed=0;
#define CHECK(c) do{if(!(c)){printf("FAIL %s L%d\n",#c,__LINE__);failed++;return;}}while(0)
#define DIM 4
int main(void){
    printf("=== Hyperbolic GNN Tests ===\n\n");
    double c=1.0;

    printf("  g1_exp_log_roundtrip...");
    {
        double v[DIM]={0,0.5,0.3,0.2};
        double p[DIM],back[DIM];
        wubu_hgnn_exp_origin(v,p,DIM,c);
        wubu_hgnn_log_origin(p,back,DIM,c);
        
        for(int i=0;i<DIM;i++)
            CHECK(fabs(v[i]-back[i])<1e-6);
    }
    printf("PASS\n");passed++;

    printf("  g2_on_hyperboloid...");
    {
        /* exp map output must satisfy L(p,p) = -1/c = -1 */
        double v[DIM]={0,0.8,-0.3,0.5};
        double p[DIM];
        wubu_hgnn_exp_origin(v,p,DIM,c);
        
        double lp=wubu_hgnn_lorentz_dot(p,p,DIM);
        printf("[L(p,p)=%.6f (want -1)] ",lp);
        CHECK(fabs(lp-(-1.0))<0.01);
    }
    printf("PASS\n");passed++;

    printf("  g3_distance_properties...");
    {
        double a[4]={1,0,0,0};  /* origin on hyperboloid */
        double v[4]={0,0.5,0,0};
        double b[4];
        wubu_hgnn_exp_origin(v,b,DIM,c);
        
        double d_self=wubu_hgnn_distance(a,a,DIM);
        CHECK(d_self<1e-6); /* distance to self = 0 */
        
        double d_ab=wubu_hgnn_distance(a,b,DIM);
        CHECK(d_ab>0); /* different points → positive distance */
        
        double d_ba=wubu_hgnn_distance(b,a,DIM);
        CHECK(fabs(d_ab-d_ba)<1e-10); /* symmetric */
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d/%d ===\n",passed,passed+failed);
    return failed>0;
}
