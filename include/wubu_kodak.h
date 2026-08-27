/* GAP-E003: Zephyr-HD Kodak — audio STFT <-> image layout */
#ifndef WUBU_KODAK_H
#define WUBU_KODAK_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int width,height;
    float mag_scale;  /* log-domain normalizer stored by pack() */
    float* pixels;  /* [H,W,3]: R=log-magnitude (normalized), G=cos(ph)/2+.5, B=sin(ph)/2+.5 */
} WubuKodak;
int  wubu_kodak_init(WubuKodak* kd,int width,int height);
void wubu_kodak_free(WubuKodak* kd);
int  wubu_kodak_pack(WubuKodak* kd,const float* frames,int num_frames,int bins);
int  wubu_kodak_unpack(const WubuKodak* kd,float* frames,int num_frames,int bins);
#ifdef __cplusplus
}
#endif
#endif
