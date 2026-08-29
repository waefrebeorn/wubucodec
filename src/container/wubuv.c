/* wubuv.c — WUBV Container Implementation */
#define _POSIX_C_SOURCE 200809L
#include "wubuv.h"
#include <stdlib.h>
#include <string.h>

/* CRC32 */
uint32_t wubuv_crc32(const uint8_t* data, size_t len) {
    static uint32_t table[256];
    static int init = 0;
    if (!init) {
        for (unsigned i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++)
                c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        init = 1;
    }
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++)
        c = table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

/* ===== Writer ===== */

WubuvWriter* wubuv_writer_open(const char* path, uint16_t width, uint16_t height,
                                uint16_t fps, uint16_t frame_count,
                                int has_audio, uint32_t audio_rate,
                                uint16_t audio_channels) {
    WubuvWriter* w = calloc(1, sizeof(WubuvWriter));
    if (!w) return NULL;
    
    w->f = fopen(path, "wb+");
    if (!w->f) { free(w); return NULL; }
    
    memcpy(w->hdr.magic, WUBUV_MAGIC, 4);
    w->hdr.version = WUBUV_VERSION;
    w->hdr.flags = 0;
    if (has_audio) w->hdr.flags |= WUBUV_FLAG_HAS_AUDIO;
    w->hdr.width = width;
    w->hdr.height = height;
    w->hdr.fps = fps;
    w->hdr.frame_count = frame_count;
    w->hdr.audio_rate = audio_rate;
    w->hdr.audio_channels = audio_channels;
    w->hdr.total_audio_samples = 0;
    w->hdr.index_offset = 0;
    
    fwrite(&w->hdr, sizeof(WubuvHeader), 1, w->f);
    
    w->index_capacity = frame_count + 100;
    w->index = calloc(w->index_capacity, sizeof(WubuvIndexEntry));
    w->index_count = 0;
    
    return w;
}

int wubuv_write_video(WubuvWriter* w, const uint8_t* compressed_data, uint32_t compressed_size, int is_key) {
    if (!w || !w->f) return -1;
    
    /* Record position for index */
    if (w->index_count >= w->index_capacity) {
        w->index_capacity *= 2;
        w->index = realloc(w->index, w->index_capacity * sizeof(WubuvIndexEntry));
    }
    
    w->index[w->index_count].frame_number = w->index_count;
    w->index[w->index_count].file_offset = ftell(w->f);
    w->index[w->index_count].frame_type = is_key ? WUBUV_FRAME_KEY : WUBUV_FRAME_INTER;
    w->index_count++;
    
    /* Write frame header */
    uint8_t type = is_key ? WUBUV_FRAME_KEY : WUBUV_FRAME_INTER;
    uint32_t payload_size = 4 + compressed_size; /* 4 bytes header + compressed data */
    
    fputc(type, w->f);
    fwrite(&payload_size, 4, 1, w->f);
    
    /* Write video payload header */
    fwrite(&w->hdr.width, 2, 1, w->f);
    fwrite(&w->hdr.height, 2, 1, w->f);
    
    /* Write compressed data */
    fwrite(compressed_data, 1, compressed_size, w->f);
    
    return 0;
}

int wubuv_write_audio(WubuvWriter* w, const float* samples, uint16_t num_samples) {
    if (!w || !w->f) return -1;
    
    /* Record position for index */
    if (w->index_count >= w->index_capacity) {
        w->index_capacity *= 2;
        w->index = realloc(w->index, w->index_capacity * sizeof(WubuvIndexEntry));
    }
    
    w->index[w->index_count].frame_number = w->index_count;
    w->index[w->index_count].file_offset = ftell(w->f);
    w->index[w->index_count].frame_type = WUBUV_FRAME_AUDIO;
    w->index_count++;
    
    /* Write frame header */
    uint8_t type = WUBUV_FRAME_AUDIO;
    uint32_t payload_size = 2 + num_samples * sizeof(float); /* 2 bytes num_samples + data */
    
    fputc(type, w->f);
    fwrite(&payload_size, 4, 1, w->f);
    
    /* Write audio payload */
    fwrite(&num_samples, 2, 1, w->f);
    fwrite(samples, sizeof(float), num_samples, w->f);
    
    w->hdr.total_audio_samples += num_samples;
    
    return 0;
}

int wubuv_writer_close(WubuvWriter* w) {
    if (!w) return -1;
    
    /* Write seek index if requested */
    if (w->hdr.flags & WUBUV_FLAG_HAS_INDEX) {
        w->hdr.index_offset = ftell(w->f);
        
        /* Write index marker */
        uint8_t type = WUBUV_FRAME_INDEX;
        uint32_t payload_size = w->index_count * sizeof(WubuvIndexEntry);
        fputc(type, w->f);
        fwrite(&payload_size, 4, 1, w->f);
        fwrite(w->index, sizeof(WubuvIndexEntry), w->index_count, w->f);
    }
    
    /* Update header with final counts (before CRC) */
    fseek(w->f, 0, SEEK_SET);
    fwrite(&w->hdr, sizeof(WubuvHeader), 1, w->f);
    
    /* Write CRC (last thing, covers everything before it) */
    fflush(w->f);
    fseek(w->f, 0, SEEK_END);
    long end = ftell(w->f);
    uint8_t* all = malloc(end);
    fseek(w->f, 0, SEEK_SET);
    fread(all, 1, end, w->f);
    uint32_t crc = wubuv_crc32(all, end);
    free(all);
    fwrite(&crc, 4, 1, w->f);
    
    fclose(w->f);
    free(w->index);
    free(w);
    
    return 0;
}

/* ===== Reader ===== */

WubuvReader* wubuv_reader_open(const char* path) {
    WubuvReader* r = calloc(1, sizeof(WubuvReader));
    if (!r) return NULL;
    
    r->f = fopen(path, "rb");
    if (!r->f) { free(r); return NULL; }
    
    if (fread(&r->hdr, sizeof(WubuvHeader), 1, r->f) != 1) {
        fclose(r->f); free(r); return NULL;
    }
    
    /* Validate */
    if (memcmp(r->hdr.magic, WUBUV_MAGIC, 4) != 0 || r->hdr.version != WUBUV_VERSION) {
        fclose(r->f); free(r); return NULL;
    }
    
    /* Read seek index if present */
    if (r->hdr.index_offset > 0) {
        fseek(r->f, r->hdr.index_offset, SEEK_SET);
        
        uint8_t type;
        uint32_t size;
        type = fgetc(r->f);
        fread(&size, 4, 1, r->f);
        
        if (type == WUBUV_FRAME_INDEX) {
            r->index_size = size / sizeof(WubuvIndexEntry);
            r->index = malloc(size);
            fread(r->index, 1, size, r->f);
        }
        
        /* Seek back to first frame */
        fseek(r->f, sizeof(WubuvHeader), SEEK_SET);
    }
    
    r->current_frame = 0;
    return r;
}

int wubuv_read_frame(WubuvReader* r, uint8_t* type_out, uint8_t** data_out,
                      uint32_t* size_out) {
    if (!r || !r->f) return -1;
    
    int type = fgetc(r->f);
    if (type == EOF) return -1;
    
    uint32_t size;
    if (fread(&size, 4, 1, r->f) != 1) return -1;
    
    uint8_t* data = malloc(size);
    if (fread(data, 1, size, r->f) != size) {
        free(data);
        return -1;
    }
    
    *type_out = (uint8_t)type;
    *data_out = data;
    *size_out = size;
    r->current_frame++;
    
    return 0;
}

int wubuv_seek_to_frame(WubuvReader* r, uint32_t frame_number) {
    if (!r) return -1;
    
    /* Use index if available */
    if (r->index && frame_number < r->index_size) {
        fseek(r->f, r->index[frame_number].file_offset, SEEK_SET);
        r->current_frame = frame_number;
        return 0;
    }
    
    /* Otherwise seek to beginning and read sequentially */
    fseek(r->f, sizeof(WubuvHeader), SEEK_SET);
    r->current_frame = 0;
    
    while (r->current_frame < frame_number) {
        uint8_t type;
        uint32_t size;
        if (wubuv_read_frame(r, &type, &(uint8_t*){NULL}, &size) != 0) return -1;
    }
    
    return 0;
}

void wubuv_reader_close(WubuvReader* r) {
    if (!r) return;
    if (r->f) fclose(r->f);
    if (r->index) free(r->index);
    free(r);
}

/* ===== Utility ===== */

int wubuv_verify(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    
    fseek(f, 0, SEEK_END);
    long end = ftell(f);
    if (end < (long)sizeof(WubuvHeader) + 4) { fclose(f); return 0; }
    
    fseek(f, 0, SEEK_SET);
    uint8_t* all = malloc(end - 4);
    fread(all, 1, end - 4, f);
    uint32_t stored;
    fread(&stored, 4, 1, f);
    fclose(f);
    
    uint32_t actual = wubuv_crc32(all, end - 4);
    free(all);
    
    return stored == actual;
}

void wubuv_print_info(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); return; }
    
    WubuvHeader hdr;
    if (fread(&hdr, sizeof(WubuvHeader), 1, f) != 1) {
        fclose(f);
        printf("Cannot read header\n");
        return;
    }
    
    fclose(f);
    
    printf("WUBV Container: %s\n", path);
    printf("  Version: %d\n", hdr.version);
    printf("  Resolution: %dx%d\n", hdr.width, hdr.height);
    printf("  FPS: %d\n", hdr.fps);
    printf("  Frames: %d\n", hdr.frame_count);
    printf("  Audio: %s\n", (hdr.flags & WUBUV_FLAG_HAS_AUDIO) ? "yes" : "no");
    if (hdr.flags & WUBUV_FLAG_HAS_AUDIO) {
        printf("  Audio rate: %d Hz\n", hdr.audio_rate);
        printf("  Audio channels: %d\n", hdr.audio_channels);
        printf("  Total audio samples: %d\n", hdr.total_audio_samples);
    }
    printf("  Has index: %s\n", (hdr.flags & WUBUV_FLAG_HAS_INDEX) ? "yes" : "no");
    
    /* Verify CRC */
    printf("  CRC: %s\n", wubuv_verify(path) ? "valid" : "INVALID");
}
