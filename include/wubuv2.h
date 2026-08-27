/* wubuv2.h -- .WUBV v2 entropy coder */
#ifndef WUBUV2_H
#define WUBUV2_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
size_t wubuv2_compress_frame(const unsigned char* Y,int W,int H,
                              unsigned char** out_buf);
#ifdef __cplusplus
}
#endif
#endif
