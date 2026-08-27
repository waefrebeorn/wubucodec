/* GROUP 18: NAL unit parsing + MP4 box structure */
#ifndef WUBU_NAL_H
#define WUBU_NAL_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const uint8_t* data;
    long size;
    int type; /* NAL unit type from header */
} WubuNalUnit;

const char* wubu_nal_type_name(int type);
long wubu_nal_find_start_code(const uint8_t* buf,long size,long offset,
                               int* start_code_len);
int  wubu_nal_parse_annexb(const uint8_t* stream,long size,
                             WubuNalUnit* units,int max_units);
long wubu_nal_remove_ep(const uint8_t* src,uint8_t* dst,long size);
long wubu_nal_add_ep(const uint8_t* src,uint8_t* dst,long size);
long wubu_annexb_to_avcc(const uint8_t* annexb,long annexb_size,
                           uint8_t* avcc_out,long avcc_cap);
void wubu_box_write_header(uint8_t* buf,long* pos,const char* type,long content_size);
long wubu_mp4_ftyp(uint8_t* buf,long cap);
#ifdef __cplusplus
}
#endif
#endif
