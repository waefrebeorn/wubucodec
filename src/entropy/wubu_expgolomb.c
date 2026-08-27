#include "wubu_expgolomb2.h"
#include <stdlib.h>
#include <string.h>

static void bw_bit(EG2_BW*b,int bit){
    size_t byte=b->pos/8;int bit_idx=7-(b->pos%8);
    if(byte<b->cap&&bit)b->buf[byte]|=(1<<bit_idx);
    b->pos++;
}

static void bw_ue(EG2_BW*b,unsigned k){
    unsigned val=k+1;
    int nbits=0;
    unsigned tmp=val;
    while(tmp>1){tmp>>=1;nbits++;}
    for(int i=0;i<nbits;i++)bw_bit(b,0);
    bw_bit(b,1);
    for(int i=nbits-1;i>=0;i--)bw_bit(b,(int)((val>>i)&1));
}

static unsigned se_to_ue(int v){return v>0?(unsigned)(v*2-1):(unsigned)(-v*2);}
static int ue_to_se(unsigned u){return (u&1)?(int)((u+1)/2):(int)(-(int)(u/2));}

void wubu_eg2_write_coeff(EG2_BW*bw,int value){
    bw_ue(bw,(unsigned)se_to_ue(value));
}

int wubu_eg2_read_coeff(const uint8_t* buf,size_t* pos,size_t cap){
    int zeros=0;
    while(*pos<cap*8){
        size_t byte=*pos/8;int bit=7-(*pos%8);
        if((buf[byte]>>bit)&1)break;
        zeros++;(*pos)++;
    }
    if(*pos>=cap*8)return 0;
    (*pos)++;
    unsigned val=1;
    for(int i=0;i<zeros;i++){
        if(*pos>=cap*8)return 0;
        size_t byte=*pos/8;int bit=7-(*pos%8);
        val=(val<<1)|((uint8_t)((buf[byte]>>bit)&1));
        (*pos)++;
    }
    return ue_to_se(val-1);
}

long wubu_eg2_encode_array(const int* values,int n,uint8_t** out_buf){
    size_t cap=(size_t)n*8;
    uint8_t* buf=calloc(cap,1);
    EG2_BW bw={buf,cap,0};
    for(int i=0;i<n;i++)wubu_eg2_write_coeff(&bw,values[i]);
    size_t actual=bw.pos/8+(bw.pos%8?1:0);
    *out_buf=buf;
    return (long)actual;
}

int wubu_eg2_decode_array(const uint8_t* buf,long buf_size,
                           int* values,int max_values){
    size_t pos=0;
    int count=0;
    while(count<max_values&&pos<(size_t)buf_size*8){
        values[count]=wubu_eg2_read_coeff(buf,&pos,(size_t)buf_size);
        count++;
    }
    return count;
}
