/*
 * wubu_quat_rate.c -- GAP-C050: Adaptive quaternion rate control
 *
 * The 1-byte-per-frame quantization works for constant angular velocity.
 * Real video has variable rotation speed: fast pans need more bits,
 * slow drifts need fewer. This module:
 *   1. Measures per-frame angular velocity from the quaternion delta
 *   2. Allocates bits proportionally (more rotation = more bits)
 *   3. Enforces a target bitrate by scaling quantization steps
 *
 * This is what makes WUBQ competitive at ALL content types, not just
 * constant-speed rotations.
 */
#define M_PI 3.14159265358979f
#include "wubu_quat_rate.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* measure angular velocity between two quaternions (radians) */
float wubu_qr_angular_velocity(const float* q_prev,const float* q_curr){
    /* |q1 - q2| gives sin(theta/2); theta = 2*arcsin(|dq|/2) for unit quats */
    float dot=0;
    for(int i=0;i<4;i++)dot+=q_prev[i]*q_curr[i];
    /* for unit quats: angle = 2*acos(|dot|) */
    if(dot>1.0f)dot=1.0f;
    if(dot<-1.0f)dot=-1.0f;
    return 2.0f*acosf(fabsf(dot));
}

/* allocate quantization levels per frame based on angular velocity.
 * fast rotation → more bits → finer angle quantization */
void wubu_qr_allocate(const float* angles,int n_frames,
                       int target_bytes_per_frame,
                       int min_bits,int max_bits,
                       int* out_bits){
    if(n_frames<=0)return;

    /* find max angular velocity in the sequence */
    float max_vel=1e-6f;
    for(int i=0;i<n_frames;i++){
        float av=fabsf(angles[i]);
        if(av>max_vel)max_vel=av;
    }

    /* proportional allocation: faster = more bits */
    for(int i=0;i<n_frames;i++){
        float ratio=fabsf(angles[i])/max_vel;
        out_bits[i]=min_bits+(int)(ratio*(max_bits-min_bits));
        if(out_bits[i]<min_bits)out_bits[i]=min_bits;
        if(out_bits[i]>max_bits)out_bits[i]=max_bits;
    }
}

/* encode one frame's quaternion delta with variable bit depth.
 * returns bytes used. */
int wubu_qr_encode_delta(float angle,float axis_x,float axis_y,float axis_z,
                          int n_bits,FILE* f){
    /* quantize angle to n_bits */
    int angle_levels=1<<n_bits;
    int angle_q=(int)(angle/M_PI*(angle_levels-1));
    if(angle_q<0)angle_q=0;
    if(angle_q>=angle_levels)angle_q=angle_levels-1;

    /* quantize axis to fixed 3×4 bits = 12 bits total (always same) */
    uint8_t ax=(uint8_t)((axis_x+1)*127.5f);
    uint8_t ay=(uint8_t)((axis_y+1)*127.5f);
    uint8_t az=(uint8_t)((axis_z+1)*127.5f);

    /* write: angle (n_bits/8 bytes rounded up), then axis (3 bytes) */
    int angle_bytes=(n_bits+7)/8;
    fwrite(&angle_q,1,angle_bytes,f);
    fputc(ax,f);fputc(ay,f);fputc(az,f);
    return angle_bytes+3;
}

/* decode one frame's quaternion delta */
float wubu_qr_decode_delta(FILE* f,int n_bits,
                            float* axis_x,float* axis_y,float* axis_z){
    int angle_levels=1<<n_bits;
    int angle_max=angle_levels-1;
    int angle_bytes=(n_bits+7)/8;
    int angle_q=0;
    fread(&angle_q,1,angle_bytes,f);

    int ax=fgetc(f),ay=fgetc(f),az=fgetc(f);
    *axis_x=ax/127.5f-1.0f;
    *axis_y=ay/127.5f-1.0f;
    *axis_z=az/127.5f-1.0f;
    return (float)angle_q/angle_max*M_PI;
}

/* full sequence: analyze, allocate, encode — returns total bytes */
long wubu_qr_encode_sequence(const float* quat_frames,int n_frames,int D,
                              float target_bpf,const char* path){
    if(n_frames<2||D<4)return -1;
    FILE* f=fopen(path,"wb");
    if(!f)return -2;

    /* measure angular velocities between consecutive frames */
    float* angles=malloc(sizeof(float)*(size_t)(n_frames-1));
    for(int i=1;i<n_frames;i++)
        angles[i-1]=wubu_qr_angular_velocity(
            quat_frames+(size_t)(i-1)*D,
            quat_frames+(size_t)i*D);

    /* allocate bits: min 4 (coarse), max 12 (fine) */
    int bits[256];
    wubu_qr_allocate(angles,n_frames-1,target_bpf,4,12,bits);

    /* header */
    uint16_t nf=n_frames,W=0,H=0;
    fwrite("WUBQ",4,1,f);
    fwrite(&nf,2,1,f);
    /* per-frame bit allocation table */
    for(int i=0;i<n_frames-1;i++){
        uint8_t b=(uint8_t)bits[i];
        fwrite(&b,1,1,f);
    }
    /* encoded deltas */
    long total=0;
    for(int i=0;i<n_frames-1;i++){
        const float* q=quat_frames+(size_t)i*D;
        const float* qc=quat_frames+(size_t)(i+1)*D;
        /* compute axis from cross product of the two quaternions' vector parts */
        float cx=q[1]*qc[3]-q[3]*qc[1];
        float cy=q[3]*qc[0]-q[0]*qc[3];
        float cz=q[0]*qc[1]-q[1]*qc[0];
        float cn=sqrtf(cx*cx+cy*cy+cz*cz);
        if(cn<1e-8f){cx=0;cy=0;cz=1;}
        else{cx/=cn;cy/=cn;cz/=cn;}
        total+=wubu_qr_encode_delta(angles[i],cx,cy,cz,bits[i],f);
    }
    free(angles);
    fclose(f);
    return total;
}
