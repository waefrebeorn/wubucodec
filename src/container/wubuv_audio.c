/* wubuv_audio.c -- WUBV container with Kodak audio support */
#include "wubuv_audio.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- Audio encode/decode via Kodak ---- */

int wubuv_audio_encode_kodak(const float* samples, int num_samples,
                              uint8_t** out, uint32_t* out_size,
                              WubuStft* stft, WubuKodak* kodak) {
    int frames;
    int bins = stft->frame_size / 2 + 1;
    
    /* Forward STFT */
    float* S = wubu_stft_forward(stft, samples, num_samples, &frames);
    if(!S) return -1;
    
    /* Pack into image */
    if(wubu_kodak_pack(kodak, S, frames, bins) != 0) {
        free(S);
        return -2;
    }
    
    /* Copy packed data */
    int img_size = kodak->width * kodak->height * 3;
    *out = malloc(img_size);
    if(!*out) {
        free(S);
        return -3;
    }
    memcpy(*out, kodak->pixels, img_size);
    *out_size = img_size;
    
    free(S);
    return 0;
}

int wubuv_audio_decode_kodak(const uint8_t* packed, uint32_t packed_size,
                              float* samples, int max_samples,
                              WubuStft* stft, WubuKodak* kodak) {
    int bins = stft->frame_size / 2 + 1;
    int frames = wubu_stft_num_frames(max_samples, stft->frame_size, stft->hop);
    
    /* Unpack from image */
    float* S = malloc(sizeof(float) * frames * bins * 2);
    if(!S) return -1;
    
    /* Copy packed into kodak pixels */
    memcpy(kodak->pixels, packed, packed_size);
    
    if(wubu_kodak_unpack(kodak, S, frames, bins) != 0) {
        free(S);
        return -2;
    }
    
    /* Inverse STFT */
    float* y = wubu_stft_inverse(stft, S, frames, max_samples);
    if(!y) {
        free(S);
        return -3;
    }
    
    memcpy(samples, y, sizeof(float) * max_samples);
    free(y);
    free(S);
    return 0;
}

/* ---- Writer Implementation ---- */

WubuvAudioWriter* wubuv_audio_writer_open(const char* path,
    uint16_t width, uint16_t height, uint16_t fps, uint16_t frame_count,
    int has_audio, uint32_t audio_rate, uint16_t channels) {
    
    WubuvAudioWriter* w = calloc(1, sizeof(WubuvAudioWriter));
    if(!w) return NULL;
    
    /* Initialize base writer */
    WubuvHeader hdr;
    wubuv_hdr_init(&hdr, width, height, fps, frame_count, has_audio, audio_rate);
    w->base.f = fopen(path, "wb+");
    if(!w->base.f) { free(w); return NULL; }
    w->base.hdr = hdr;
    fwrite(&hdr, sizeof(WubuvHeader), 1, w->base.f);
    w->base.body_start = sizeof(WubuvHeader);
    
    /* Initialize audio */
    w->has_audio = has_audio;
    w->audio_rate = audio_rate;
    w->channels = channels;
    
    if(has_audio) {
        /* Initialize STFT: 512-point FFT, 128 hop (75% overlap) */
        if(wubu_stft_init(&w->stft, 512, 128) != 0) {
            fclose(w->base.f);
            free(w);
            return NULL;
        }
        
        /* Initialize Kodak: 256x256 image */
        if(wubu_kodak_init(&w->kodak, 256, 256) != 0) {
            wubu_stft_free(&w->stft);
            fclose(w->base.f);
            free(w);
            return NULL;
        }
        
        /* Initialize psychoacoustic model */
        if(wubu_pa_init(&w->pa, 512, audio_rate) != 0) {
            wubu_kodak_free(&w->kodak);
            wubu_stft_free(&w->stft);
            fclose(w->base.f);
            free(w);
            return NULL;
        }
        
        w->audio_buf_cap = 4096;
        w->audio_buffer = malloc(sizeof(float) * w->audio_buf_cap);
        w->audio_buf_len = 0;
    }
    
    return w;
}

int wubuv_audio_write_video(WubuvAudioWriter* w, const uint8_t* rgb, int is_inter) {
    return wubuv_write_frame(&w->base, rgb, is_inter);
}

int wubuv_audio_write_audio(WubuvAudioWriter* w, const float* samples, int num_samples) {
    if(!w->has_audio) return -1;
    
    /* Buffer audio samples */
    int needed = w->audio_buf_len + num_samples;
    if(needed > w->audio_buf_cap) {
        while(needed > w->audio_buf_cap) w->audio_buf_cap *= 2;
        w->audio_buffer = realloc(w->audio_buffer, sizeof(float) * w->audio_buf_cap);
    }
    memcpy(w->audio_buffer + w->audio_buf_len, samples, sizeof(float) * num_samples);
    w->audio_buf_len += num_samples;
    
    /* Encode when we have enough for one STFT frame */
    int frame_samples = w->stft.frame_size;
    int hop = w->stft.hop;
    
    while(w->audio_buf_len >= frame_samples) {
        /* Encode one audio frame */
        uint8_t* packed = NULL;
        uint32_t packed_size = 0;
        
        int ret = wubuv_audio_encode_kodak(w->audio_buffer, frame_samples,
                                            &packed, &packed_size,
                                            &w->stft, &w->kodak);
        if(ret != 0) return ret;
        
        /* Write audio frame to container */
        /* Frame type = AUDIO (2) */
        fputc(WUBUV_FRAME_AUDIO, w->base.f);
        /* Payload size: 4 (frame_index) + 2 (num_samples) + 2 (bins) + 4 (mag_scale) + packed_size */
        uint32_t payload_size = 4 + 2 + 2 + 4 + packed_size;
        fwrite(&payload_size, 4, 1, w->base.f);
        /* Payload */
        uint32_t frame_idx = w->base.hdr.frame_count; /* reuse as audio frame counter */
        fwrite(&frame_idx, 4, 1, w->base.f);
        uint16_t ns = frame_samples;
        fwrite(&ns, 2, 1, w->base.f);
        uint16_t b = w->stft.frame_size / 2 + 1;
        fwrite(&b, 2, 1, w->base.f);
        float ms = w->kodak.mag_scale;
        fwrite(&ms, 4, 1, w->base.f);
        fwrite(packed, 1, packed_size, w->base.f);
        
        free(packed);
        
        /* Shift buffer by hop */
        memmove(w->audio_buffer, w->audio_buffer + hop, 
                sizeof(float) * (w->audio_buf_len - hop));
        w->audio_buf_len -= hop;
    }
    
    return 0;
}

int wubuv_audio_writer_close(WubuvAudioWriter* w) {
    /* Flush remaining audio */
    if(w->has_audio && w->audio_buf_len > 0) {
        /* Pad to frame size */
        if(w->audio_buf_len < w->stft.frame_size) {
            memset(w->audio_buffer + w->audio_buf_len, 0, 
                   sizeof(float) * (w->stft.frame_size - w->audio_buf_len));
        }
        
        uint8_t* packed = NULL;
        uint32_t packed_size = 0;
        wubuv_audio_encode_kodak(w->audio_buffer, w->stft.frame_size,
                                  &packed, &packed_size,
                                  &w->stft, &w->kodak);
        
        fputc(WUBUV_FRAME_AUDIO, w->base.f);
        uint32_t payload_size = 4 + 2 + 2 + 4 + packed_size;
        fwrite(&payload_size, 4, 1, w->base.f);
        uint32_t frame_idx = 0;
        fwrite(&frame_idx, 4, 1, w->base.f);
        uint16_t ns = w->audio_buf_len;
        fwrite(&ns, 2, 1, w->base.f);
        uint16_t b = w->stft.frame_size / 2 + 1;
        fwrite(&b, 2, 1, w->base.f);
        float ms = w->kodak.mag_scale;
        fwrite(&ms, 4, 1, w->base.f);
        fwrite(packed, 1, packed_size, w->base.f);
        free(packed);
    }
    
    /* Write CRC */
    fflush(w->base.f);
    long end = ftell(w->base.f);
    uint8_t* all = malloc(end);
    fseek(w->base.f, 0, SEEK_SET);
    fread(all, 1, end, w->base.f);
    uint32_t crc = wubuv_crc32(all, end);
    free(all);
    fwrite(&crc, 4, 1, w->base.f);
    
    /* Cleanup audio resources */
    if(w->has_audio) {
        wubu_stft_free(&w->stft);
        wubu_kodak_free(&w->kodak);
        wubu_pa_free(&w->pa);
        free(w->audio_buffer);
    }
    
    /* Close file and free struct */
    if(w->base.prev_frame) {
        free(w->base.prev_frame);
        w->base.prev_frame = NULL;
    }
    fclose(w->base.f);
    free(w);
    return 0;
}

/* ---- Reader Implementation ---- */

WubuvAudioReader* wubuv_audio_reader_open(const char* path) {
    WubuvAudioReader* r = calloc(1, sizeof(WubuvAudioReader));
    if(!r) return NULL;
    
    /* Initialize base reader */
    r->base.f = fopen(path, "rb");
    if(!r->base.f) { free(r); return NULL; }
    if(fread(&r->base.hdr, sizeof(WubuvHeader), 1, r->base.f) != 1 ||
       !wubuv_hdr_valid(&r->base.hdr)) {
        fclose(r->base.f);
        free(r);
        return NULL;
    }
    
    r->base.n_read = 0;
    r->base.prev_frame = malloc(r->base.hdr.width * r->base.hdr.height * 3);
    r->base.has_prev = 0;
    
    /* Initialize audio if present */
    r->has_audio = (r->base.hdr.flags & 1) != 0;
    if(r->has_audio) {
        wubu_stft_init(&r->stft, 512, 128);
        wubu_kodak_init(&r->kodak, 256, 256);
        wubu_pa_init(&r->pa, 512, r->base.hdr.audio_rate);
        r->audio_buf_cap = 4096;
        r->audio_buffer = malloc(sizeof(float) * r->audio_buf_cap);
        r->audio_buf_len = 0;
    }
    
    return r;
}

int wubuv_audio_read_video(WubuvAudioReader* r, uint8_t* rgb_out, int* is_inter) {
    return wubuv_read_frame(&r->base, rgb_out, is_inter);
}

int wubuv_audio_read_audio(WubuvAudioReader* r, float* samples, int max_samples) {
    if(!r->has_audio) return 0;
    
    /* Read next audio frame from file */
    int type = fgetc(r->base.f);
    if(type == EOF) return 0;
    
    if(type != WUBUV_FRAME_AUDIO) {
        /* Not an audio frame, seek back */
        fseek(r->base.f, -1, SEEK_CUR);
        return 0;
    }
    
    uint32_t payload_size;
    if(fread(&payload_size, 4, 1, r->base.f) != 1) return 0;
    
    /* Read payload */
    uint32_t frame_idx;
    uint16_t num_samples, bins;
    float mag_scale;
    fread(&frame_idx, 4, 1, r->base.f);
    fread(&num_samples, 2, 1, r->base.f);
    fread(&bins, 2, 1, r->base.f);
    fread(&mag_scale, 4, 1, r->base.f);
    
    uint32_t packed_size = payload_size - 12;
    uint8_t* packed = malloc(packed_size);
    fread(packed, 1, packed_size, r->base.f);
    
    /* Decode */
    r->kodak.mag_scale = mag_scale;
    int decoded = max_samples;
    wubuv_audio_decode_kodak(packed, packed_size, samples, decoded, &r->stft, &r->kodak);
    
    free(packed);
    return decoded;
}

int wubuv_audio_reader_close(WubuvAudioReader* r) {
    if(r->has_audio) {
        wubu_stft_free(&r->stft);
        wubu_kodak_free(&r->kodak);
        wubu_pa_free(&r->pa);
        free(r->audio_buffer);
    }
    wubuv_reader_close(&r->base);
    free(r);
    return 0;
}
