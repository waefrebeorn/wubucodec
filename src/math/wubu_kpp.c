/*
 * wubu_kpp.c -- GAP-D033: Hyperbolic k-means++ seeding (D² sampling)
 *
 * Research source: Arthur & Vassilvitskii 2007 (k-means++). The D²
 * seeding picks the first center arbitrarily, then each next center
 * with probability proportional to squared distance to the nearest
 * existing center — spreading seeds across the manifold.
 *
 * Hyperbolic version: distances are geodesic; the O(log k) competitive
 * bound carries over since the Poincaré ball with geodesic distance is
 * still a proper metric space. This upgrades C020's random init.
 */
#include "wubu_kpp.h"
#include <stdlib.h>
#include <math.h>

static float kp_dist(const float* a,const float* b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){
        float df=a[d]-b[d];ab2+=df*df;
        a2+=a[d]*a[d];b2+=b[d]*b[d];
    }
    float den=(1-c*a2)*(1-c*b2);
    if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

int wubu_kpp_seed(const float* pts,int n,int D,int k,float c,
                   unsigned* seed,int* out_centers){
    if(!pts||k<1||k>n||!out_centers)return -1;

    /* nearest-center distance squared per point */
    float* d2=malloc(sizeof(float)*(size_t)n);
    if(!d2)return -2;
    for(int i=0;i<n;i++)d2[i]=1e30f;

    /* first center: pseudo-random */
    *seed=*seed*1103515245u+12345u;
    int first=(*seed>>16)%n;
    out_centers[0]=first;
    for(int i=0;i<n;i++)
        d2[i]=kp_dist(pts+(size_t)i*D,pts+(size_t)first*D,D,c);

    for(int m=1;m<k;m++){
        /* total D2 weight (squared geodesic) */
        double total=0;
        for(int i=0;i<n;i++)total+=(double)d2[i]*d2[i];
        if(total<=0){
            /* degenerate: all points coincide — pick arbitrary remaining */
            out_centers[m]=(out_centers[m-1]+1)%n;
            continue;
        }
        /* sample proportional to d2 */
        *seed=*seed*1103515245u+12345u;
        double target=((double)((*seed>>8)%100000)/100000.0)*total;
        double acc=0;
        int chosen=n-1;
        for(int i=0;i<n;i++){
            acc+=(double)d2[i]*d2[i];
            if(acc>=target){chosen=i;break;}
        }
        out_centers[m]=chosen;
        /* update nearest distances */
        for(int i=0;i<n;i++){
            float d=kp_dist(pts+(size_t)i*D,pts+(size_t)chosen*D,D,c);
            if(d<d2[i])d2[i]=d;
        }
    }
    free(d2);
    return 0;
}
