#include "wubu_zigrle.h"
#include <stdlib.h>
#include <string.h>

static const int zigzag_order[64]={
     0, 1, 8,16, 9, 2, 3,10,
    17,24,32,25,18,11, 4, 5,
    12,19,26,33,40,48,41,34,
    27,20,13, 6, 7,14,21,28,
    35,42,49,56,57,50,43,36,
    29,22,15,23,30,37,44,51,
    58,59,52,45,38,31,39,46,
    53,60,61,54,47,55,62,63
};

static int inv_zz_table[64];
static int inv_zz_init=0;

void wubu_zz_scan(const int* block,int* scanned){
    for(int i=0;i<64;i++)scanned[i]=block[zigzag_order[i]];
}

void wubu_zz_unscan(const int* scanned,int* block){
    if(!inv_zz_init){
        for(int i=0;i<64;i++)inv_zz_table[zigzag_order[i]]=i;
        inv_zz_init=1;
    }
    for(int i=0;i<64;i++)block[i]=scanned[inv_zz_table[i]];
}

int wubu_rle_encode(const int* scanned,int n,int* runs,int* values){
    int count=0,run=0;
    for(int i=0;i<n;i++){
        if(scanned[i]==0){run++;}
        else{
            while(run>63&&count<n){runs[count]=63;values[count]=0;count++;run-=64;}
            if(count<n){runs[count]=run;values[count]=scanned[i];count++;}
            run=0;
        }
    }
    if(run>0&&count<n){runs[count]=0;values[count]=0;count++;}
    return count;
}

int wubu_rle_decode(const int* runs,const int* values,int n_pairs,int* scanned,int n){
    memset(scanned,0,sizeof(int)*(size_t)n);
    int pos=0;
    for(int i=0;i<n_pairs;i++){
        if(runs[i]==0&&values[i]==0)break;
        pos+=runs[i];
        if(pos<n)scanned[pos]=values[i];
        pos++;
    }
    return pos;
}
