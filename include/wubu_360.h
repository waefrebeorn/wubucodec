/* GROUP 23: 360° video & VR */
#include <stdint.h>
#ifndef WUBU_360_H
#define WUBU_360_H
#ifdef __cplusplus
extern "C" {
#endif
void wubu_erp_to_sphere(int x,int y,int W,int H,double* lon,double* lat);
void wubu_sphere_to_erp(double lon,double lat,int W,int H,int* x,int* y);
int  wubu_cmp_face(double dx,double dy,double dz,int* fx,int* fy,int face_size);
long wubu_pad_erp(const uint8_t* src,uint8_t* padded,int W,int H,int pad);
double wubu_viewport_fraction(double fov_h_degrees,double fov_v_degrees,
                                double center_lat_degrees);
int  wubu_me_wraparound_adjust(int dx,int ref_width);
#ifdef __cplusplus
}
#endif
#endif
