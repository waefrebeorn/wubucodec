/* .WUBV — the WuBu video container format */
#ifndef WUBUV_H
#define WUBUV_H
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    char     magic[4];    /* "WUBV" */
    uint16_t version;     /* 1 */
    uint16_t flags;       /* bit0: has_audio */
    uint16_t width;
    uint16_t height;
    uint16_t fps;
    uint16_t frame_count;
    uint32_t audio_rate;
} WubuvHeader;

void wubuv_hdr_init(WubuvHeader* h,uint16_t w,uint16_t hgt,
                    uint16_t fps,uint16_t frames,int has_audio,
                    uint32_t audio_rate);
int  wubuv_hdr_valid(const WubuvHeader* h);
uint32_t wubuv_crc32(const uint8_t* data,size_t len);

typedef struct {
    FILE* f;
    WubuvHeader hdr;
    long body_start;
    uint8_t* prev_frame;
    int has_prev;
} WubuvWriter;

WubuvWriter* wubuv_writer_open(const char* path,const WubuvHeader* h);
int  wubuv_write_frame(WubuvWriter* w,const uint8_t* rgb,int is_inter);
int  wubuv_writer_close(WubuvWriter* w);

typedef struct {
    FILE* f;
    WubuvHeader hdr;
    int n_read;
    uint8_t* prev_frame;
    int has_prev;
} WubuvReader;

WubuvReader* wubuv_reader_open(const char* path);
int  wubuv_read_frame(WubuvReader* r,uint8_t* rgb_out,int* is_inter);
void wubuv_reader_close(WubuvReader* r);
int  wubuv_verify(const char* path);
#ifdef __cplusplus
}
#endif
#endif
