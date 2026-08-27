/*
 * wubu_nal.c -- GROUP 18: NAL unit parsing + MP4 box structure
 *
 * G18.09: NAL unit parsing/writing layer
 * G18.10: Annex B ↔ MP4 (AVCC) format conversion
 * G18.01: Basic MP4 box writer
 */
#include "wubu_nal.h"
#include <stdlib.h>
#include <string.h>

/* ===== NAL Unit Types ===== */

const char* wubu_nal_type_name(int type){
    switch(type){
        case 1:return "coded slice non-IDR";
        case 2:case 3:case 4:return "coded slice partition";
        case 5:return "coded slice IDR";
        case 6:return "SEI";
        case 7:return "SPS";
        case 8:return "PPS";
        case 9:return "access unit delimiter";
        default:return "unknown";
    }
}

/* ===== Annex B Parsing ===== */

/* Find the next start code in the buffer starting from offset.
 * Returns byte offset of start code, or -1 if not found.
 * Sets *start_code_len to 3 or 4 depending on which variant found. */
long wubu_nal_find_start_code(const uint8_t* buf,long size,long offset,
                               int* start_code_len){
    for(long i=offset;i+2<size;i++){
        if(buf[i]==0&&buf[i+1]==0){
            if(buf[i+2]==1){*start_code_len=3;return i;}
            if(i+3<size&&buf[i+2]==0&&buf[i+3]==1){*start_code_len=4;return i;}
        }
    }
    return -1;
}

/* Extract all NAL units from an Annex B stream */
int wubu_nal_parse_annexb(const uint8_t* stream,long size,
                            WubuNalUnit* units,int max_units){
    int count=0;
    long pos=0;
    
    while(pos<size&&count<max_units){
        int sc_len;
        long nal_start=wubu_nal_find_start_code(stream,size,pos,&sc_len);
        if(nal_start<0)break;
        
        pos=nal_start+sc_len;
        long nal_data_start=pos;
        
        /* find next start code or end of stream */
        int next_sc_len;
        long next_sc=wubu_nal_find_start_code(stream,size,pos,&next_sc_len);
        long nal_end=(next_sc>=0)?next_sc:size;
        
        /* strip trailing zeros before next start code */
        while(nal_end>nal_data_start&&stream[nal_end-1]==0)nal_end--;
        
        if(nal_end>nal_data_start){
            units[count].data=stream+nal_data_start;
            units[count].size=nal_end-nal_data_start;
            units[count].type=stream[nal_data_start]&0x1F;
            count++;
        }
        pos=nal_end;
    }
    return count;
}

/* ===== Emulation Prevention ===== */

/*
 * In RBSP (Raw Byte Sequence Payload), sequences of 0x000000-0x000003 
 * are escaped by inserting 0x03 after the second zero byte.
 * EPB (emulation prevention byte): 0x000000 → 0x00000300, etc.
 */

/* Remove emulation prevention bytes from NAL payload */
long wubu_nal_remove_ep(const uint8_t* src,uint8_t* dst,long size){
    long dst_pos=0;
    int zeros=0;
    
    for(long i=0;i<size;i++){
        if(zeros==2&&src[i]==0x03){
            /* skip this emulation prevention byte */
            zeros=0;
            continue;
        }
        if(src[i]==0x00)zeros++;
        else zeros=0;
        
        dst[dst_pos++]=src[i];
    }
    return dst_pos;
}

/* Add emulation prevention bytes to raw payload */
long wubu_nal_add_ep(const uint8_t* src,uint8_t* dst,long size){
    long dst_pos=0;
    int zeros=0;
    
    for(long i=0;i<size;i++){
        if(zeros==2&&(src[i]<=0x03)){
            /* insert EPB before this byte */
            dst[dst_pos++]=0x03;
            zeros=0;
        }
        if(src[i]==0x00)zeros++;
        else zeros=0;
        
        dst[dst_pos++]=src[i];
    }
    return dst_pos;
}

/* ===== Annex B ↔ AVCC Conversion ===== */

/* Convert Annex B to AVCC: replace start codes with 4-byte length prefixes,
 * extract SPS/PPS into separate output */
long wubu_annexb_to_avcc(const uint8_t* annexb,long annexb_size,
                           uint8_t* avcc_out,long avcc_cap){
    WubuNalUnit units[64];
    int n_units=wubu_nal_parse_annexb(annexb,annexb_size,units,64);
    
    long out_pos=0;
    for(int i=0;i<n_units;i++){
        uint32_t len=(uint32_t)units[i].size;
        /* write big-endian length prefix */
        if(out_pos+4+(long)len<=avcc_cap){
            avcc_out[out_pos++]=(len>>24)&0xFF;
            avcc_out[out_pos++]=(len>>16)&0xFF;
            avcc_out[out_pos++]=(len>>8)&0xFF;
            avcc_out[out_pos++]=len&0xFF;
            memcpy(avcc_out+out_pos,units[i].data,units[i].size);
            out_pos+=units[i].size;
        }
    }
    return out_pos;
}

/* ===== Simple MP4 Box Writer ===== */

void wubu_box_write_header(uint8_t* buf,long* pos,const char* type,long content_size){
    uint32_t total=(uint32_t)(content_size+8);
    buf[(*pos)++]=(total>>24)&0xFF;
    buf[(*pos)++]=(total>>16)&0xFF;
    buf[(*pos)++]=(total>>8)&0xFF;
    buf[(*pos)++]=total&0xFF;
    memcpy(buf+(*pos),type,4);
    (*pos)+=4;
}

/* create a minimal ftyp box */
long wubu_mp4_ftyp(uint8_t* buf,long cap){
    long pos=0;
    if(cap<20)return -1;
    wubu_box_write_header(buf,&pos,"ftyp",12);
    memcpy(buf+pos,"isom",4);pos+=4;
    buf[pos++]=0;buf[pos++]=0;buf[pos++]=2;buf[pos++]=0; /* minor version 512 */
    memcpy(buf+pos,"isom",4);pos+=4;
    return pos;
}
