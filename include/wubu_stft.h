/*
 * wubu_stft.h -- STFT/ISTFT in C11 (GAP-E001) — the Kodak's spectral engine
 */
#ifndef WUBU_STFT_H
#define WUBU_STFT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int frame_size;   /* FFT size (power of 2, >=8)  */
    int hop;          /* samples between frames      */
    float* window;    /* periodic Hann               */
} WubuStft;

int  wubu_stft_init(WubuStft* st,int frame_size,int hop);
void wubu_stft_free(WubuStft* st);

int wubu_stft_num_frames(int signal_len,int frame_size,int hop);

/* x[len] -> [num_frames][F/2+1] interleaved (re,im). Caller frees. */
float* wubu_stft_forward(const WubuStft* st,const float* x,int len,int* out_frames);

/* frames -> y[out_len] via weighted overlap-add. Caller frees. */
float* wubu_stft_inverse(const WubuStft* st,const float* frames,int num_frames,int out_len);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_STFT_H */
