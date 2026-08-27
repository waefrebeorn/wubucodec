/* GAP-E005: audio-conditioned video fidelity weighting */
#ifndef WUBU_AV_FIDELITY_H
#define WUBU_AV_FIDELITY_H
#ifdef __cplusplus
extern "C" {
#endif

/* band_energy_norm: [T,5] normalized band energies (from wubu_bands) */
void wubu_av_weights(const float* band_energy_norm,float* weights,int T,
                     float alpha,float beta,float w_min);

/* weighted fidelity + rate loss */
float wubu_av_fidelity_loss(const float* video_dist,const float* weights,
                            int T,float bits,float lambda);
#ifdef __cplusplus
}
#endif
#endif
