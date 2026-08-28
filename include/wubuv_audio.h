/* wubuv_audio.h -- WUBV container audio extension
 *
 * Adds Kodak audio support to the WUBV container.
 * Audio is encoded using the Zephyr-HD reversible STFT codec:
 *   - Forward STFT → log-magnitude + phase → image-like layout
 *   - Pack into sidecar frames interleaved with video
 *   - Psychoacoustic masking for perceptual bit allocation
 *
 * WUBV v2 frame types:
 *   0 = KEY (intra video)
 *   1 = INTER (delta video)
 *   2 = AUDIO (Kodak spectral frame)
 */
#ifndef WUBUV_AUDIO_H
#define WUBUV_AUDIO_H
#include "wubuv.h"  /* WubuvWriter, WubuvReader, WubuvHeader */
#include "wubu_kodak.h"
#include "wubu_stft.h"
#include "wubu_psychoacoustic.h"
#include <stdint.h>

#define WUBUV_FRAME_KEY   0
#define WUBUV_FRAME_INTER 1
#define WUBUV_FRAME_AUDIO 2

/* Audio frame: Kodak-packed spectral data */
typedef struct {
    uint32_t frame_index;     /* audio frame index */
    uint16_t num_samples;     /* original samples */
    uint16_t bins;            /* FFT bins */
    float mag_scale;          /* log-domain normalizer */
    uint8_t* packed;          /* [H*W*3] RGB-packed spectrum */
    uint32_t packed_size;     /* bytes */
} WubuvAudioFrame;

/* Extended writer with audio */
typedef struct {
    WubuvWriter base;
    WubuStft stft;
    WubuKodak kodak;
    WubuPsychoacoustic pa;
    int has_audio;
    uint32_t audio_rate;
    uint16_t channels;
    float* audio_buffer;
    int audio_buf_len;
    int audio_buf_cap;
} WubuvAudioWriter;

/* Extended reader with audio */
typedef struct {
    WubuvReader base;
    WubuStft stft;
    WubuKodak kodak;
    WubuPsychoacoustic pa;
    int has_audio;
    float* audio_buffer;
    int audio_buf_len;
    int audio_buf_cap;
} WubuvAudioReader;

/* Writer API */
WubuvAudioWriter* wubuv_audio_writer_open(const char* path, 
    uint16_t width, uint16_t height, uint16_t fps, uint16_t frame_count,
    int has_audio, uint32_t audio_rate, uint16_t channels);

int wubuv_audio_write_video(WubuvAudioWriter* w, const uint8_t* rgb, int is_inter);
int wubuv_audio_write_audio(WubuvAudioWriter* w, const float* samples, int num_samples);
int wubuv_audio_writer_close(WubuvAudioWriter* w);

/* Reader API */
WubuvAudioReader* wubuv_audio_reader_open(const char* path);
int wubuv_audio_read_video(WubuvAudioReader* r, uint8_t* rgb_out, int* is_inter);
int wubuv_audio_read_audio(WubuvAudioReader* r, float* samples, int max_samples);
int wubuv_audio_reader_close(WubuvAudioReader* w);

/* Utility */
int wubuv_audio_encode_kodak(const float* samples, int num_samples,
                              uint8_t** out, uint32_t* out_size,
                              WubuStft* stft, WubuKodak* kodak);
int wubuv_audio_decode_kodak(const uint8_t* packed, uint32_t packed_size,
                              float* samples, int max_samples,
                              WubuStft* stft, WubuKodak* kodak);

#endif
