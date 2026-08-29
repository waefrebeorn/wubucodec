/* container_check.c — Triple-check WUBV container format robustness
 *
 * Tests:
 * 1. Header validation
 * 2. Frame write/read roundtrip
 * 3. CRC integrity
 * 4. Audio frame interleaving
 * 5. Corruption detection
 * 6. Multiple resolutions
 * 7. Edge cases (empty frames, max size, etc.)
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ===== WUBV Container Format ===== */

#define WUBUV_FRAME_KEY   0
#define WUBUV_FRAME_INTER 1
#define WUBUV_FRAME_AUDIO 2

#pragma pack(push, 1)
typedef struct {
    char     magic[4];    /* "WUBV" */
    uint16_t version;     /* 2 */
    uint16_t flags;       /* bit0: has_audio */
    uint16_t width;
    uint16_t height;
    uint16_t fps;
    uint16_t frame_count;
    uint32_t audio_rate;
} WubuvHeader;
#pragma pack(pop)

typedef struct {
    FILE* f;
    WubuvHeader hdr;
    long body_start;
    uint8_t* prev_frame;
    int has_prev;
} WubuvWriter;

typedef struct {
    FILE* f;
    WubuvHeader hdr;
    int n_read;
    uint8_t* prev_frame;
    int has_prev;
} WubuvReader;

/* CRC32 */
static uint32_t crc_table[256];
static int crc_init = 0;

static void crc_setup(void) {
    if (crc_init) return;
    for (unsigned i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++)
            c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crc_table[i] = c;
    }
    crc_init = 1;
}

static uint32_t crc32(const uint8_t* data, size_t len) {
    crc_setup();
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++)
        c = crc_table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

/* ===== Writer ===== */

static WubuvWriter* wubuv_writer_open(const char* path, uint16_t w, uint16_t h,
                                       uint16_t fps, uint16_t frame_count,
                                       int has_audio, uint32_t audio_rate) {
    WubuvWriter* wr = calloc(1, sizeof(WubuvWriter));
    if (!wr) return NULL;
    
    wr->f = fopen(path, "wb+");
    if (!wr->f) { free(wr); return NULL; }
    
    memcpy(wr->hdr.magic, "WUBV", 4);
    wr->hdr.version = 2;
    wr->hdr.flags = has_audio ? 1 : 0;
    wr->hdr.width = w;
    wr->hdr.height = h;
    wr->hdr.fps = fps;
    wr->hdr.frame_count = frame_count;
    wr->hdr.audio_rate = audio_rate;
    
    fwrite(&wr->hdr, sizeof(WubuvHeader), 1, wr->f);
    wr->body_start = sizeof(WubuvHeader);
    
    return wr;
}

static int wubuv_write_video(WubuvWriter* wr, const uint8_t* y_data,
                              uint16_t width, uint16_t height, int is_key) {
    /* Frame: type(1) + size(4) + data(width*height) */
    uint8_t type = is_key ? WUBUV_FRAME_KEY : WUBUV_FRAME_INTER;
    uint32_t size = width * height;
    
    fputc(type, wr->f);
    fwrite(&size, 4, 1, wr->f);
    fwrite(y_data, 1, size, wr->f);
    
    return 0;
}

static int wubuv_write_audio(WubuvWriter* wr, const float* samples,
                              uint16_t num_samples) {
    /* Audio frame: type(1) + size(4) + data */
    uint8_t type = WUBUV_FRAME_AUDIO;
    uint32_t size = num_samples * sizeof(float);
    
    fputc(type, wr->f);
    fwrite(&size, 4, 1, wr->f);
    fwrite(samples, 1, size, wr->f);
    
    return 0;
}

static int wubuv_writer_close(WubuvWriter* wr) {
    /* Write CRC */
    fflush(wr->f);
    long end = ftell(wr->f);
    uint8_t* all = malloc(end);
    fseek(wr->f, 0, SEEK_SET);
    fread(all, 1, end, wr->f);
    uint32_t crc = crc32(all, end);
    free(all);
    fwrite(&crc, 4, 1, wr->f);
    
    fclose(wr->f);
    free(wr);
    return 0;
}

/* ===== Reader ===== */

static WubuvReader* wubuv_reader_open(const char* path) {
    WubuvReader* r = calloc(1, sizeof(WubuvReader));
    if (!r) return NULL;
    
    r->f = fopen(path, "rb");
    if (!r->f) { free(r); return NULL; }
    
    if (fread(&r->hdr, sizeof(WubuvHeader), 1, r->f) != 1) {
        fclose(r->f); free(r); return NULL;
    }
    
    /* Validate */
    if (memcmp(r->hdr.magic, "WUBV", 4) != 0 || r->hdr.version != 2) {
        fclose(r->f); free(r); return NULL;
    }
    
    r->n_read = 0;
    return r;
}

static int wubuv_read_frame(WubuvReader* r, uint8_t* type_out, uint8_t** data_out,
                             uint32_t* size_out) {
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
    r->n_read++;
    
    return 0;
}

static int wubuv_verify(const char* path) {
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
    
    uint32_t actual = crc32(all, end - 4);
    free(all);
    
    return stored == actual;
}

static void wubuv_reader_close(WubuvReader* r) {
    fclose(r->f);
    free(r);
}

/* ===== Tests ===== */

static int test_header_validation(void) {
    printf("=== Test 1: Header Validation ===\n");
    
    int pass = 1;
    
    /* Valid header */
    WubuvWriter* wr = wubuv_writer_open("/tmp/test1.wubv", 320, 240, 30, 100, 1, 48000);
    if (!wr) {
        printf("  FAIL: Cannot create writer\n");
        return 0;
    }
    wubuv_writer_close(wr);
    
    WubuvReader* r = wubuv_reader_open("/tmp/test1.wubv");
    if (!r) {
        printf("  FAIL: Cannot open valid file\n");
        return 0;
    }
    
    if (r->hdr.width != 320 || r->hdr.height != 240 || r->hdr.fps != 30) {
        printf("  FAIL: Header mismatch\n");
        pass = 0;
    } else {
        printf("  PASS: Header valid (320x240, 30fps, audio=48kHz)\n");
    }
    wubuv_reader_close(r);
    
    /* Invalid magic */
    FILE* f = fopen("/tmp/test_bad.wubv", "wb");
    fwrite("BAD\x00", 1, 4, f);
    fclose(f);
    r = wubuv_reader_open("/tmp/test_bad.wubv");
    if (r) {
        printf("  FAIL: Accepted bad magic\n");
        pass = 0;
        wubuv_reader_close(r);
    } else {
        printf("  PASS: Rejected bad magic\n");
    }
    
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

static int test_frame_roundtrip(void) {
    printf("=== Test 2: Frame Write/Read Roundtrip ===\n");
    
    int pass = 1;
    uint16_t W = 320, H = 240;
    
    WubuvWriter* wr = wubuv_writer_open("/tmp/test2.wubv", W, H, 30, 2, 0, 0);
    if (!wr) return 0;
    
    /* Write key frame */
    uint8_t frame1[320 * 240];
    for (int i = 0; i < 320 * 240; i++) frame1[i] = (uint8_t)(i % 256);
    wubuv_write_video(wr, frame1, W, H, 1);
    
    /* Write inter frame */
    uint8_t frame2[320 * 240];
    for (int i = 0; i < 320 * 240; i++) frame2[i] = (uint8_t)((i + 128) % 256);
    wubuv_write_video(wr, frame2, W, H, 0);
    
    wubuv_writer_close(wr);
    
    /* Read back */
    WubuvReader* r = wubuv_reader_open("/tmp/test2.wubv");
    if (!r) return 0;
    
    uint8_t type;
    uint8_t* data;
    uint32_t size;
    
    /* Read frame 1 */
    if (wubuv_read_frame(r, &type, &data, &size) != 0) {
        printf("  FAIL: Cannot read frame 1\n");
        pass = 0;
    } else if (type != WUBUV_FRAME_KEY || size != W * H) {
        printf("  FAIL: Frame 1 type=%d, size=%d\n", type, size);
        pass = 0;
    } else if (memcmp(data, frame1, W * H) != 0) {
        printf("  FAIL: Frame 1 data mismatch\n");
        pass = 0;
    } else {
        printf("  PASS: Frame 1 roundtrip OK\n");
    }
    free(data);
    
    /* Read frame 2 */
    if (wubuv_read_frame(r, &type, &data, &size) != 0) {
        printf("  FAIL: Cannot read frame 2\n");
        pass = 0;
    } else if (type != WUBUV_FRAME_INTER || size != W * H) {
        printf("  FAIL: Frame 2 type=%d, size=%d\n", type, size);
        pass = 0;
    } else if (memcmp(data, frame2, W * H) != 0) {
        printf("  FAIL: Frame 2 data mismatch\n");
        pass = 0;
    } else {
        printf("  PASS: Frame 2 roundtrip OK\n");
    }
    free(data);
    
    wubuv_reader_close(r);
    
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

static int test_crc_integrity(void) {
    printf("=== Test 3: CRC Integrity ===\n");
    
    int pass = 1;
    uint16_t W = 64, H = 48;
    
    WubuvWriter* wr = wubuv_writer_open("/tmp/test3.wubv", W, H, 30, 1, 0, 0);
    uint8_t frame[64 * 48];
    memset(frame, 0xAB, sizeof(frame));
    wubuv_write_video(wr, frame, W, H, 1);
    wubuv_writer_close(wr);
    
    /* Verify CRC */
    if (wubuv_verify("/tmp/test3.wubv")) {
        printf("  PASS: CRC valid\n");
    } else {
        printf("  FAIL: CRC invalid\n");
        pass = 0;
    }
    
    /* Corrupt a byte */
    FILE* f = fopen("/tmp/test3.wubv", "r+b");
    fseek(f, sizeof(WubuvHeader) + 10, SEEK_SET);
    uint8_t bad;
    fread(&bad, 1, 1, f);
    fseek(f, sizeof(WubuvHeader) + 10, SEEK_SET);
    fputc(bad ^ 0xFF, f);
    fclose(f);
    
    if (!wubuv_verify("/tmp/test3.wubv")) {
        printf("  PASS: Corruption detected\n");
    } else {
        printf("  FAIL: Corruption not detected\n");
        pass = 0;
    }
    
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

static int test_audio_interleave(void) {
    printf("=== Test 4: Audio/Video Interleave ===\n");
    
    int pass = 1;
    uint16_t W = 320, H = 240;
    
    WubuvWriter* wr = wubuv_writer_open("/tmp/test4.wubv", W, H, 30, 2, 1, 48000);
    
    /* Write: video, audio, video, audio */
    uint8_t frame1[320 * 240], frame2[320 * 240];
    memset(frame1, 100, sizeof(frame1));
    memset(frame2, 200, sizeof(frame2));
    
    float audio1[4800], audio2[4800];
    for (int i = 0; i < 4800; i++) {
        audio1[i] = sinf(2 * 3.14159f * 440 * i / 48000);
        audio2[i] = sinf(2 * 3.14159f * 880 * i / 48000);
    }
    
    wubuv_write_video(wr, frame1, W, H, 1);
    wubuv_write_audio(wr, audio1, 4800);
    wubuv_write_video(wr, frame2, W, H, 0);
    wubuv_write_audio(wr, audio2, 4800);
    
    wubuv_writer_close(wr);
    
    /* Read back */
    WubuvReader* r = wubuv_reader_open("/tmp/test4.wubv");
    uint8_t type;
    uint8_t* data;
    uint32_t size;
    
    /* Frame 1: video */
    wubuv_read_frame(r, &type, &data, &size);
    if (type != WUBUV_FRAME_KEY) { printf("  FAIL: Expected KEY, got %d\n", type); pass = 0; }
    else if (memcmp(data, frame1, W * H) != 0) { printf("  FAIL: Video 1 mismatch\n"); pass = 0; }
    else printf("  PASS: Video 1 OK\n");
    free(data);
    
    /* Frame 2: audio */
    wubuv_read_frame(r, &type, &data, &size);
    if (type != WUBUV_FRAME_AUDIO) { printf("  FAIL: Expected AUDIO, got %d\n", type); pass = 0; }
    else if (size != 4800 * sizeof(float)) { printf("  FAIL: Audio 1 size %d\n", size); pass = 0; }
    else printf("  PASS: Audio 1 OK\n");
    free(data);
    
    /* Frame 3: video */
    wubuv_read_frame(r, &type, &data, &size);
    if (type != WUBUV_FRAME_INTER) { printf("  FAIL: Expected INTER, got %d\n", type); pass = 0; }
    else if (memcmp(data, frame2, W * H) != 0) { printf("  FAIL: Video 2 mismatch\n"); pass = 0; }
    else printf("  PASS: Video 2 OK\n");
    free(data);
    
    /* Frame 4: audio */
    wubuv_read_frame(r, &type, &data, &size);
    if (type != WUBUV_FRAME_AUDIO) { printf("  FAIL: Expected AUDIO, got %d\n", type); pass = 0; }
    else printf("  PASS: Audio 2 OK\n");
    free(data);
    
    wubuv_reader_close(r);
    
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

static int test_multiple_resolutions(void) {
    printf("=== Test 5: Multiple Resolutions ===\n");
    
    int pass = 1;
    uint16_t resolutions[][2] = {
        {160, 120}, {320, 240}, {640, 480}, {800, 600}, {1024, 768}, {1920, 1080}
    };
    
    for (int r = 0; r < 6; r++) {
        uint16_t W = resolutions[r][0], H = resolutions[r][1];
        char path[256];
        snprintf(path, sizeof(path), "/tmp/test_res_%dx%d.wubv", W, H);
        
        WubuvWriter* wr = wubuv_writer_open(path, W, H, 30, 1, 0, 0);
        if (!wr) {
            printf("  FAIL: Cannot create writer for %dx%d\n", W, H);
            pass = 0;
            continue;
        }
        
        uint8_t* frame = malloc(W * H);
        memset(frame, 0xAA, W * H);
        wubuv_write_video(wr, frame, W, H, 1);
        wubuv_writer_close(wr);
        
        /* Read back */
        WubuvReader* reader = wubuv_reader_open(path);
        if (!reader) {
            printf("  FAIL: Cannot read %dx%d\n", W, H);
            pass = 0;
            free(frame);
            continue;
        }
        
        uint8_t type;
        uint8_t* data;
        uint32_t size;
        wubuv_read_frame(reader, &type, &data, &size);
        
        if (size != W * H) {
            printf("  FAIL: %dx%d size mismatch (%d vs %d)\n", W, H, size, W * H);
            pass = 0;
        } else {
            printf("  PASS: %dx%d OK\n", W, H);
        }
        
        free(data);
        free(frame);
        wubuv_reader_close(reader);
    }
    
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

static int test_edge_cases(void) {
    printf("=== Test 6: Edge Cases ===\n");
    
    int pass = 1;
    
    /* Empty frame (all zeros) */
    WubuvWriter* wr = wubuv_writer_open("/tmp/test_empty.wubv", 64, 48, 30, 1, 0, 0);
    uint8_t empty[64 * 48];
    memset(empty, 0, sizeof(empty));
    wubuv_write_video(wr, empty, 64, 48, 1);
    wubuv_writer_close(wr);
    
    WubuvReader* r = wubuv_reader_open("/tmp/test_empty.wubv");
    uint8_t type;
    uint8_t* data;
    uint32_t size;
    wubuv_read_frame(r, &type, &data, &size);
    if (memcmp(data, empty, 64 * 48) != 0) {
        printf("  FAIL: Empty frame mismatch\n");
        pass = 0;
    } else {
        printf("  PASS: Empty frame OK\n");
    }
    free(data);
    wubuv_reader_close(r);
    
    /* Maximum values */
    wr = wubuv_writer_open("/tmp/test_max.wubv", 64, 48, 30, 1, 0, 0);
    uint8_t max_frame[64 * 48];
    memset(max_frame, 0xFF, sizeof(max_frame));
    wubuv_write_video(wr, max_frame, 64, 48, 1);
    wubuv_writer_close(wr);
    
    r = wubuv_reader_open("/tmp/test_max.wubv");
    wubuv_read_frame(r, &type, &data, &size);
    if (memcmp(data, max_frame, 64 * 48) != 0) {
        printf("  FAIL: Max frame mismatch\n");
        pass = 0;
    } else {
        printf("  PASS: Max frame OK\n");
    }
    free(data);
    wubuv_reader_close(r);
    
    /* No audio flag */
    wr = wubuv_writer_open("/tmp/test_noaudio.wubv", 64, 48, 30, 1, 0, 0);
    wubuv_write_video(wr, empty, 64, 48, 1);
    wubuv_writer_close(wr);
    
    r = wubuv_reader_open("/tmp/test_noaudio.wubv");
    if (r->hdr.flags & 1) {
        printf("  FAIL: No audio flag set incorrectly\n");
        pass = 0;
    } else {
        printf("  PASS: No audio flag correct\n");
    }
    wubuv_reader_close(r);
    
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

/* ===== Main ===== */

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║          CONTAINER FORMAT — Triple Check                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    int pass = 0, total = 0;
    
    total++; pass += test_header_validation();
    total++; pass += test_frame_roundtrip();
    total++; pass += test_crc_integrity();
    total++; pass += test_audio_interleave();
    total++; pass += test_multiple_resolutions();
    total++; pass += test_edge_cases();
    
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  RESULTS: %d/%d tests passed", pass, total);
    if (pass == total) {
        printf(" — ALL PASS ✓\n");
    } else {
        printf(" — %d FAILURES ✗\n", total - pass);
    }
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    return (pass == total) ? 0 : 1;
}
