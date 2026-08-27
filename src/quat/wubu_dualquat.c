/*
 * wubu_dualquat.c -- GAP-C062: Dual quaternion motion model
 * (rotation + translation in ONE 8-element representation)
 *
 * Research source: arXiv:2310.07623 — dual quaternions jointly represent
 * rotation AND translation. The real part is the rotation quaternion,
 * the dual part encodes translation. This gives the codec a SINGLE
 * primitive for both camera pan (rotation) and dolly (translation).
 *
 * Dual number: z = a + εb where ε² = 0
 * Dual quaternion: Q = Qr + εQd where Qr = rotation, Qd = ½t⊗Qr
 */
#define M_PI 3.14159265358979f
#include "wubu_dualquat.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* create dual quat from axis-angle rotation + 3D translation */
void wubu_dq_create(float angle,float ax,float ay,float az,
                     float tx,float ty,float tz,float* out){
    /* real part: unit rotation quaternion */
    float half=angle/2;
    out[0]=cosf(half);
    out[1]=sinf(half)*ax;
    out[2]=sinf(half)*ay;
    out[3]=sinf(half)*az;

    /* dual part: 0.5 * t ⊗ Qr (translation embedded) */
    float t[4]={0,tx,ty,tz};
    float qr[4]={out[0],out[1],out[2],out[3]};
    /* Hamilton product t ⊗ qr */
    float tq[4];
    tq[0]=t[0]*qr[0]-t[1]*qr[1]-t[2]*qr[2]-t[3]*qr[3];
    tq[1]=t[0]*qr[1]+t[1]*qr[0]+t[2]*qr[3]-t[3]*qr[2];
    tq[2]=t[0]*qr[2]-t[1]*qr[3]+t[2]*qr[0]+t[3]*qr[1];
    tq[3]=t[0]*qr[3]+t[1]*qr[2]-t[2]*qr[1]+t[3]*qr[0];
    for(int i=0;i<4;i++)out[4+i]=0.5f*tq[i];
}

/* extract translation from a dual quaternion */
void wubu_dq_get_translation(const float* dq,float* t){
    /* t = 2 * Qd ⊗ conj(Qr) */
    float qr_inv[4]={dq[0],-dq[1],-dq[2],-dq[3]};
    float qd[4]={dq[4],dq[5],dq[6],dq[7]};
    /* qd ⊗ qr_inv */
    float prod[4];
    prod[0]=qd[0]*qr_inv[0]-qd[1]*qr_inv[1]-qd[2]*qr_inv[2]-qd[3]*qr_inv[3];
    prod[1]=qd[0]*qr_inv[1]+qd[1]*qr_inv[0]+qd[2]*qr_inv[3]-qd[3]*qr_inv[2];
    prod[2]=qd[0]*qr_inv[2]-qd[1]*qr_inv[3]+qd[2]*qr_inv[0]+qd[3]*qr_inv[1];
    prod[3]=qd[0]*qr_inv[3]+qd[1]*qr_inv[2]-qd[2]*qr_inv[1]+qd[3]*qr_inv[0];
    t[0]=2*prod[1];t[1]=2*prod[2];t[2]=2*prod[3];
}

/* rigid transform: rotate + translate a 3D point using dual quaternion */
void wubu_dq_transform(const float* dq,const float* point,float* result){
    /* result = Qr ⊗ p ⊗ conj(Qr) + t */
    float qr[4]={dq[0],dq[1],dq[2],dq[3]};
    float qc[4]={dq[0],-dq[1],-dq[2],-dq[3]};
    float p[4]={0,point[0],point[1],point[2]};

    /* p' = qr ⊗ p ⊗ qc */
    float qp[4];
    qp[0]=qr[0]*p[0]-qr[1]*p[1]-qr[2]*p[2]-qr[3]*p[3];
    qp[1]=qr[0]*p[1]+qr[1]*p[0]+qr[2]*p[3]-qr[3]*p[2];
    qp[2]=qr[0]*p[2]-qr[1]*p[3]+qr[2]*p[0]+qr[3]*p[1];
    qp[3]=qr[0]*p[3]+qr[1]*p[2]-qr[2]*p[1]+qr[3]*p[0];

    result[0]=qp[0]*qc[0]-qp[1]*qc[1]-qp[2]*qc[2]-qp[3]*qc[3];
    result[1]=qp[0]*qc[1]+qp[1]*qc[0]-qp[2]*qc[3]+qp[3]*qc[2];
    result[2]=qp[0]*qc[2]-qp[1]*qc[3]+qp[2]*qc[0]+qp[3]*qc[1];

    /* add translation */
    float t[3];
    wubu_dq_get_translation(dq,t);
    result[0]+=t[0];result[1]+=t[1];result[2]+=t[2];
}

/* SLERP between two dual quaternions (linear blend approximation) */
void wubu_dq_slerp(const float* qa,const float* qb,float t,float* out){
    /* normalize both, then linear interpolate (approximation of ScLERP) */
    float na[8],nb[8];
    for(int i=0;i<4;i++){
        na[i]=qa[i]/sqrtf(qa[0]*qa[0]+qa[1]*qa[1]+qa[2]*qa[2]+qa[3]*qa[3]);
        nb[i]=qb[i]/sqrtf(qb[0]*qb[0]+qb[1]*qb[1]+qb[2]*qb[2]+qb[3]*qb[3]);
    }
    for(int i=0;i<8;i++)
        out[i]=(1-t)*na[i]+t*nb[i];

    /* re-normalize real part */
    float rn=sqrtf(out[0]*out[0]+out[1]*out[1]+out[2]*out[2]+out[3]*out[3]);
    if(rn>1e-10f)
        for(int i=0;i<4;i++)out[i]/=rn;
}
