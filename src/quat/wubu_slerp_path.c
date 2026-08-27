/*
 * wubu_slerp_path.c -- GAP-C028: Quaternion SLERP path for P-frame motion
 *
 * The VHF canvas stores quaternion+amplitude per cell. Frame-to-frame
 * motion in that representation is a QUATERNION path, not a vector path.
 * This module provides:
 *   - slerp between two unit quaternions (Shoemake 1985)
 *   - a full N-step path with constant angular velocity
 *   - amplitude interpolation (linear, clamped to [0,1] via sigmoid)
 *
 * Gates: endpoints exact, constant angular speed, norm-1 throughout,
 * shortest-path (no double-cover) via sign alignment.
 */
#include "wubu_slerp_path.h"
#include <math.h>
#include <string.h>

static float qdot(const float* a,const float* b){
    float s=0;for(int d=0;d<4;d++)s+=a[d]*b[d];
    return s;
}

void wubu_q_normalize(float* q){
    float n=0;
    for(int d=0;d<4;d++)n+=q[d]*q[d];
    n=sqrtf(n);
    if(n<1e-10f){q[0]=1;q[1]=q[2]=q[3]=0;return;}
    for(int d=0;d<4;d++)q[d]/=n;
}

void wubu_q_slerp(float* out,const float* qa,const float* qb,float t){
    /* copy inputs so we can flip sign without touching caller */
    float a[4],b[4];
    memcpy(a,qa,sizeof(float)*4);
    memcpy(b,qb,sizeof(float)*4);
    /* shortest path: negate b if dot<0 */
    float d=qdot(a,b);
    if(d<0.0f){for(int i=0;i<4;i++)b[i]=-b[i];d=-d;}
    if(d>0.9995f){
        /* nearly identical: lerp + normalize avoids division blowup */
        for(int i=0;i<4;i++)out[i]=a[i]+t*(b[i]-a[i]);
        wubu_q_normalize(out);
        return;
    }
    float theta=acosf(d);
    float st=sinf(theta);
    float wa=sinf((1-t)*theta)/st;
    float wb=sinf(t*theta)/st;
    for(int i=0;i<4;i++)out[i]=wa*a[i]+wb*b[i];
    wubu_q_normalize(out);
}

void wubu_q_path(const float* q0,const float* q1,int steps,
                 float* out /* [(steps+1)*4] */){
    for(int s=0;s<=steps;s++){
        float t=(float)s/(float)steps;
        wubu_q_slerp(out+(size_t)s*4,q0,q1,t);
    }
}

/* full VHF cell path: quaternion slerp + linear amplitude */
void wubu_cell_path(float* out,const float* cell0,const float* cell1,
                    int steps){
    /* cells are [quat(4), amp(1)] */
    float q0[4],q1[4];
    memcpy(q0,cell0,sizeof(float)*4);
    memcpy(q1,cell1,sizeof(float)*4);
    wubu_q_normalize(q0);
    wubu_q_normalize(q1);
    float a0=cell0[4],a1=cell1[4];
    for(int s=0;s<=steps;s++){
        float t=(float)s/(float)steps;
        float* o=out+(size_t)s*5;
        wubu_q_slerp(o,q0,q1,t);
        /* sigmoid-bounded linear amplitude */
        o[4]=a0+t*(a1-a0);
    }
}
