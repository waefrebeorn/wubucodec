/* GAP-E002: perceptual-band split over STFT bins (Kodak 5-band layout) */
#ifndef WUBU_BANDS_H
#define WUBU_BANDS_H
#include "wubu_stft.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int start[5];   /* inclusive bin start per band */
    int end[5];     /* exclusive bin end            */
} WubuBandTable;

/* Bass | Mids | Presence | Treble | Harmonics — edges from 48k/Fsr grid */
int  wubu_band_split(const WubuStft* st, WubuBandTable* bt);
void wubu_band_energy(const WubuBandTable* bt,const float* frames,
                      int num_frames,int bins,float* out /*[5]*/);
void wubu_band_normalize(float* e5);
#ifdef __cplusplus
}
#endif
#endif
