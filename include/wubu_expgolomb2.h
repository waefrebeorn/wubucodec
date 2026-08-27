#ifndef WUBU_EXPGOLOMB2_H
#define WUBU_EXPGOLOMB2_H
#include <stdint.h>
#include <stddef.h>
typedef struct{uint8_t*buf;size_t cap;size_t pos;}EG2_BW;
void wubu_eg2_write_coeff(EG2_BW*bw,int value);
int  wubu_eg2_read_coeff(const uint8_t* buf,size_t* pos,size_t cap);
long wubu_eg2_encode_array(const int* values,int n,uint8_t** out_buf);
int  wubu_eg2_decode_array(const uint8_t* buf,long buf_size,
                            int* values,int max_values);
#endif
