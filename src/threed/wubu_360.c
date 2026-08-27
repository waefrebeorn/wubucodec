/*
 * wubu_360.c -- GROUP 23: 360° video & VR
 *
 * G23.01: Equirectangular projection (ERP)
 * G23.02: Cubemap projection (CMP) face packing
 * G23.04: Spherical rotation-aware motion estimation
 * G23.05: Wrap-around padding for horizontal boundaries
 * G23.06: Viewport fraction computation
 */
#define M_PI 3.14159265358979f
#include "wubu_360.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== G23.01: Equirectangular Projection ===== */

/*
 * ERP maps spherical coords to a rectangle:
 *   x = W * (λ + π) / (2π)
 *   y = H * (φ + π/2) / π
 * where λ = longitude [-π,π], φ = latitude [-π/2,π/2]
 *
 * Inverse: given pixel (x,y), compute spherical direction.
 */
void wubu_erp_to_sphere(int x,int y,int W,int H,
                          double* lon,double* lat){
    *lon=((double)x/W)*2*M_PI-M_PI;
    *lat=M_PI/2-((double)y/H)*M_PI;
}

void wubu_sphere_to_erp(double lon,double lat,int W,int H,
                          int* x,int* y){
    *x=(int)(((lon+M_PI)/(2*M_PI))*W)%W;
    if(*x<0)*x+=W;
    *y=(int)((1-(lat/M_PI+0.5))*H);
    if(*y<0)*y=0;if(*y>=H)*y=H-1;
}

/* ===== G23.02: Cubemap Projection ===== */

/*
 * CMP: sphere → 6 cube faces (front/right/back/left/top/bottom)
 * Each face is W/4 × H/3 in a 3×2 grid layout:
 *   layout = [     ] [ top ] [     ]
 *            [left] [front] [right] [back]
 *   face_size = W/4
 */

/* determine which cube face and local coordinates for a 3D direction */
int wubu_cmp_face(double dx,double dy,double dz,
                    int* fx,int* fy,int face_size){
    double abs_x=fabs(dx),abs_y=fabs(dy),abs_z=fabs(dz);
    
    /* find dominant axis */
    int face;
    double ua,va; /* local coordinates */
    
    if(abs_x>=abs_y&&abs_x>=abs_z){
        /* left or right face */
        if(dx>0){face=2;ua=-dz/abs_x;va=-dy/abs_x;} /* right */
        else{face=0;ua=dz/abs_x;va=-dy/abs_x;}       /* left */
    }else if(abs_y>=abs_x&&abs_y>=abs_z){
        /* top or bottom */
        if(dy>0){face=4;ua=dx/abs_y;va=dz/abs_y;}    /* top */
        else{face=5;ua=dx/abs_y;va=-dz/abs_y;}        /* bottom */
    }else{
        /* front or back */
        if(dz>0){face=1;ua=dx/abs_z;va=-dy/abs_z;}   /* front */
        else{face=3;ua=-dx/abs_z;va=-dy/abs_z;}       /* back */
    }
    
    /* map [-1,1] to [0,face_size) */
    *fx=(int)((ua+1)/2*face_size);
    *fy=(int)((va+1)/2*face_size);
    if(*fx<0)*fx=0;if(*fx>=face_size)*fx=face_size-1;
    if(*fy<0)*fy=0;if(*fy>=face_size)*fy=face_size-1;
    
    return face;
}

/* ===== G23.05: Wrap-around padding ===== */

/*
 * For ERP, the left edge wraps around to the right edge.
 * Pad horizontally by 'pad' pixels using wrap-around.
 * Also mirror vertically at poles.
 */
long wubu_pad_erp(const uint8_t* src,uint8_t* padded,
                    int W,int H,int pad){
    int pw=W+2*pad,ph=H+2*pad;
    
    for(int py=0;py<ph;py++)
        for(int px=0;px<pw;px++){
            int sx=px-pad;
            int sy=py-pad;
            
            /* horizontal wrap-around */
            sx=((sx%W)+W)%W;
            
            /* vertical mirror at poles */
            if(sy<0)sy=-sy-1;
            if(sy>=H)sy=2*H-sy-1;
            if(sy<0)sy=0;if(sy>=H)sy=H-1;
            
            padded[(size_t)py*pw+px]=src[(size_t)sy*W+sx];
        }
    return (long)pw*ph;
}

/* ===== G23.06: Viewport fraction ===== */

/* fraction of sphere visible given FOV angles and center latitude */
double wubu_viewport_fraction(double fov_h_degrees,double fov_v_degrees,
                                double center_lat_degrees){
    double theta_h=fov_h_degrees*M_PI/180;
    double theta_v=fov_v_degrees*M_PI/180;
    double phi_c=center_lat_degrees*M_PI/180;
    
    /* from the analytical formula (see arXiv 2606 network-06-00066 eq.3):
     * α = θh × (sin(φc + θv/2) - sin(φc - θv/2)) / (4π) */
    double sin_top=sin(phi_c+theta_v/2);
    double sin_bot=sin(phi_c-theta_v/2);
    
    return theta_h*(sin_top-sin_bot)/(4*M_PI);
}

/* ===== G23.04: Rotation-aware ME ===== */

/*
 * Adjust MV for horizontal wrap-around: if the best match crosses the
 * right edge, it should be found on the left side of the reference frame.
 * Returns adjusted dx that accounts for wrap-around.
 */
int wubu_me_wraparound_adjust(int dx,int ref_width){
    int half=ref_width/2;
    if(dx>half)dx-=ref_width;
    if(dx<-half)dx+=ref_width;
    return dx;
}
