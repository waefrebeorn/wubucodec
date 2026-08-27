/* GAP-E006: Psychoacoustic masking threshold model */
#ifndef WUBU_PSYCHOACOUSTIC_H
#define WUBU_PSYCHOACOUSTIC_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int fft_size;
    int sample_rate;
    float* bin_bark; /* [fft_size/2+1] precomputed bark index */
} WubuPsychoacoustic;

float wubu_pa_hz_to_bark(float hz);
int  wubu_pa_init(WubuPsychoacoustic* pa,int fft_size,int sample_rate);
void wubu_pa_free(WubuPsychoacoustic* pa);

/* masking_threshold[k]: per-bin masking level from spreading function.
 * smr_db: signal-to-mask ratio in dB (29 dB = conservative default). */
void wubu_pa_masking_threshold(const WubuPsychoacoustic* pa,
                                const float* spectrum,float* threshold,
                                float smr_db);

/* SMR per perceptual band (5 bands) — feeds AV fidelity weights. */
void wubu_pa_smr_per_band(const WubuPsychoacoustic* pa,
                          const float* spectrum,const float* threshold,
                          float* smr);

#ifdef __cplusplus
}
#endif
#endif
