/* GAP-C071: zigzag scan + RLE */
#ifndef WUBU_ZIGRLE_H
#define WUBU_ZIGRLE_H
#ifdef __cplusplus
extern "C" {
#endif
void wubu_zz_scan(const int* block,int* scanned);
void wubu_zz_unscan(const int* scanned,int* block);
int  wubu_rle_encode(const int* scanned,int n,int* runs,int* values);
int  wubu_rle_decode(const int* runs,const int* values,int n_pairs,
                      int* scanned,int n);
#ifdef __cplusplus
}
#endif
#endif
