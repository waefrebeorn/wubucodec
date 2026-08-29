/* player_check.c — Triple-check WUBV player robustness
 *
 * Tests:
 * 1. Player opens valid file
 * 2. Player reads all frame types
 * 3. Player handles audio sync
 * 4. Player handles end of file
 * 5. Player reports correct metadata
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ===== WUBV Container (same as container_check.c) ===== */

#define WUBUV_FRAME_KEY   0
#define WUBUV_FRAME_INTER 1
#define WUBUV_FRAME_AUDIO 2

#pragma pack(push, 1)
typedef struct {
    char     magic[4];
    uint16_t version;
    uint16_t flags;
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
    int n_read;
} WubuvReader;

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

typedef struct {
    FILE* f;
    WubuvHeader hdr;
} WubuvWriter;

static WubuvWriter* wubuv_writer_open(const char* path, uint16_t w, uint16_t h,
                                       uint16_t fps, uint16_t frame_count,
                                       int has_audio, uint32_t audio_rate) {
    WubuvWriter* wr = calloc(1, sizeof(WubuvWriter));
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
    return wr;
}

static int wubuv_write_video(WubuvWriter* wr, const uint8_t* y_data,
                              uint16_t width, uint16_t height, int is_key) {
    uint8_t type = is_key ? WUBUV_FRAME_KEY : WUBUV_FRAME_INTER;
    uint32_t size = width * height;
    fputc(type, wr->f);
    fwrite(&size, 4, 1, wr->f);
    fwrite(y_data, 1, size, wr->f);
    return 0;
}

static int wubuv_write_audio(WubuvWriter* wr, const float* samples,
                              uint16_t num_samples) {
    uint8_t type = WUBUV_FRAME_AUDIO;
    uint32_t size = num_samples * sizeof(float);
    fputc(type, wr->f);
    fwrite(&size, 4, 1, wr->f);
    fwrite(samples, 1, size, wr->f);
    return 0;
}

static int wubuv_writer_close(WubuvWriter* wr) {
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
    r->f = fopen(path, "rb");
    if (!r->f) { free(r); return NULL; }
    
    if (fread(&r->hdr, sizeof(WubuvHeader), 1, r->f) != 1 ||
        memcmp(r->hdr.magic, "WUBV", 4) != 0 || r->hdr.version != 2) {
        fclose(r->f); free(r); return NULL;
    }
    
    return r;
}

static int wubuv_read_frame(WubuvReader* r, uint8_t* type_out, uint8_t** data_out,
                             uint32_t* size_out) {
    int type = fgetc(r->f);
    if (type == EOF) return -1;
    
    uint32_t size;
    if (fread(&size, 4, 1, r->f) != 1) return -1;
    
    uint8_t* data = malloc(size);
    if (fread(data, 1, size, r->f) != size) { free(data); return -1; }
    
    *type_out = (uint8_t)type;
    *data_out = data;
    *size_out = size;
    r->n_read++;
    return 0;
}

static void wubuv_reader_close(WubuvReader* r) {
    fclose(r->f);
    free(r);
}

/* ===== Tests ===== */

static int test_player_open(void) {
    printf("=== Test 1: Player Opens Valid File ===\n");
    
    /* Create test file */
    WubuvWriter* wr = wubuv_writer_open("/tmp/player_test.wubv", 320, 240, 30, 2, 1, 48000);
    uint8_t frame[320 * 240];
    memset(frame, 128, sizeof(frame));
    wubuv_write_video(wr, frame, 320, 240, 1);
    float audio[4800];
    memset(audio, 0, sizeof(audio));
    wubuv_write_audio(wr, audio, 4800);
    wubuv_writer_close(wr);
    
    /* Open with player */
    WubuvReader* r = wubuv_reader_open("/tmp/player_test.wubv");
    if (!r) {
        printf("  FAIL: Cannot open file\n");
        return 0;
    }
    
    printf("  Width: %d (expected 320)\n", r->hdr.width);
    printf("  Height: %d (expected 240)\n", r->hdr.height);
    printf("  FPS: %d (expected 30)\n", r->hdr.fps);
    printf("  Audio rate: %d (expected 48000)\n", r->hdr.audio_rate);
    printf("  Has audio: %d (expected 1)\n", r->hdr.flags & 1);
    
    if (r->hdr.width == 320 && r->hdr.height == 240 && r->hdr.fps == 30 &&
        r->hdr.audio_rate == 48000 && (r->hdr.flags & 1)) {
        printf("  PASS\n\n");
        wubuv_reader_close(r);
        return 1;
    } else {
        printf("  FAIL\n\n");
        wubuv_reader_close(r);
        return 0;
    }
}

static int test_player_read_all_types(void) {
    printf("=== Test 2: Player Reads All Frame Types ===\n");
    
    WubuvWriter* wr = wubuv_writer_open("/tmp/player_test2.wubv", 64, 48, 30, 3, 1, 48000);
    uint8_t frame[64 * 48];
    memset(frame, 100, sizeof(frame));
    
    wubuv_write_video(wr, frame, 64, 48, 1);  /* KEY */
    wubuv_write_video(wr, frame, 64, 48, 0);  /* INTER */
    wubuv_write_audio(wr, (float[]){0.1f, 0.2f}, 2);  /* AUDIO */
    wubuv_write_video(wr, frame, 64, 48, 0);  /* INTER */
    
    wubuv_writer_close(wr);
    
    WubuvReader* r = wubuv_reader_open("/tmp/player_test2.wubv");
    if (!r) return 0;
    
    uint8_t type;
    uint8_t* data;
    uint32_t size;
    int pass = 1;
    
    /* Frame 1: KEY */
    wubuv_read_frame(r, &type, &data, &size);
    if (type != WUBUV_FRAME_KEY) { printf("  FAIL: Expected KEY\n"); pass = 0; }
    else printf("  PASS: Read KEY frame\n");
    free(data);
    
    /* Frame 2: INTER */
    wubuv_read_frame(r, &type, &data, &size);
    if (type != WUBUV_FRAME_INTER) { printf("  FAIL: Expected INTER\n"); pass = 0; }
    else printf("  PASS: Read INTER frame\n");
    free(data);
    
    /* Frame 3: AUDIO */
    wubuv_read_frame(r, &type, &data, &size);
    if (type != WUBUV_FRAME_AUDIO) { printf("  FAIL: Expected AUDIO\n"); pass = 0; }
    else printf("  PASS: Read AUDIO frame\n");
    free(data);
    
    /* Frame 4: INTER */
    wubuv_read_frame(r, &type, &data, &size);
    if (type != WUBUV_FRAME_INTER) { printf("  FAIL: Expected INTER\n"); pass = 0; }
    else printf("  PASS: Read INTER frame\n");
    free(data);
    
    /* End of file */
    int ret = wubuv_read_frame(r, &type, &data, &size);
    if (ret != -1) { printf("  FAIL: Expected EOF\n"); pass = 0; }
    else printf("  PASS: EOF detected\n");
    
    wubuv_reader_close(r);
    
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

static int test_player_audio_sync(void) {
    printf("=== Test 3: Audio Sync ===\n");
    
    /* Create file with known audio/video pattern */
    WubuvWriter* wr = wubuv_writer_open("/tmp/player_sync.wubv", 64, 48, 30, 4, 1, 48000);
    uint8_t frame[64 * 48];
    memset(frame, 100, sizeof(frame));
    
    /* Write 4 video frames with audio between them */
    for (int i = 0; i < 4; i++) {
        wubuv_write_video(wr, frame, 64, 48, i == 0);
        float audio[1200];
        for (int j = 0; j < 1200; j++) audio[j] = (float)i / 4.0f;
        wubuv_write_audio(wr, audio, 1200);
    }
    
    wubuv_writer_close(wr);
    
    /* Read back and verify sync */
    WubuvReader* r = wubuv_reader_open("/tmp/player_sync.wubv");
    if (!r) return 0;
    
    uint8_t type;
    uint8_t* data;
    uint32_t size;
    int pass = 1;
    
    for (int i = 0; i < 4; i++) {
        /* Read video */
        wubuv_read_frame(r, &type, &data, &size);
        if (type != (i == 0 ? WUBUV_FRAME_KEY : WUBUV_FRAME_INTER)) {
            printf("  FAIL: Video %d type mismatch\n", i);
            pass = 0;
        }
        free(data);
        
        /* Read audio */
        wubuv_read_frame(r, &type, &data, &size);
        if (type != WUBUV_FRAME_AUDIO) {
            printf("  FAIL: Audio %d type mismatch\n", i);
            pass = 0;
        } else {
            float* audio = (float*)data;
            if (audio[0] != (float)i / 4.0f) {
                printf("  FAIL: Audio %d data mismatch (expected %.2f, got %.2f)\n",
                       i, (float)i / 4.0f, audio[0]);
                pass = 0;
            }
        }
        free(data);
    }
    
    wubuv_reader_close(r);
    
    if (pass) printf("  PASS: Audio/video interleaving correct\n");
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

static int test_player_eof_handling(void) {
    printf("=== Test 4: EOF Handling ===\n");
    
    WubuvWriter* wr = wubuv_writer_open("/tmp/player_eof.wubv", 64, 48, 30, 1, 0, 0);
    uint8_t frame[64 * 48];
    memset(frame, 100, sizeof(frame));
    wubuv_write_video(wr, frame, 64, 48, 1);
    wubuv_writer_close(wr);
    
    WubuvReader* r = wubuv_reader_open("/tmp/player_eof.wubv");
    if (!r) return 0;
    
    uint8_t type;
    uint8_t* data;
    uint32_t size;
    int pass = 1;
    
    /* Read the one frame */
    int ret = wubuv_read_frame(r, &type, &data, &size);
    if (ret != 0) { printf("  FAIL: Cannot read frame\n"); pass = 0; }
    else free(data);
    
    /* Try to read past EOF */
    ret = wubuv_read_frame(r, &type, &data, &size);
    if (ret != -1) { printf("  FAIL: Expected EOF\n"); pass = 0; }
    else printf("  PASS: EOF returned correctly\n");
    
    /* Try again (should still be EOF) */
    ret = wubuv_read_frame(r, &type, &data, &size);
    if (ret != -1) { printf("  FAIL: Expected EOF on second try\n"); pass = 0; }
    else printf("  PASS: EOF consistent\n");
    
    wubuv_reader_close(r);
    
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

static int test_player_large_file(void) {
    printf("=== Test 5: Large File Handling ===\n");
    
    /* Create file with many frames */
    int num_frames = 100;
    WubuvWriter* wr = wubuv_writer_open("/tmp/player_large.wubv", 320, 240, 30, num_frames, 0, 0);
    uint8_t frame[320 * 240];
    
    for (int i = 0; i < num_frames; i++) {
        memset(frame, (uint8_t)i, sizeof(frame));
        wubuv_write_video(wr, frame, 320, 240, i == 0);
    }
    
    wubuv_writer_close(wr);
    
    WubuvReader* r = wubuv_reader_open("/tmp/player_large.wubv");
    if (!r) return 0;
    
    uint8_t type;
    uint8_t* data;
    uint32_t size;
    int pass = 1;
    int count = 0;
    
    while (wubuv_read_frame(r, &type, &data, &size) == 0) {
        if (size != 320 * 240) {
            printf("  FAIL: Frame %d size mismatch (%d vs %d)\n", count, size, 320 * 240);
            pass = 0;
        }
        free(data);
        count++;
    }
    
    if (count != num_frames) {
        printf("  FAIL: Read %d frames, expected %d\n", count, num_frames);
        pass = 0;
    } else {
        printf("  PASS: Read all %d frames\n", num_frames);
    }
    
    wubuv_reader_close(r);
    
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

/* ===== Main ===== */

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║              PLAYER — Triple Check                          ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    int pass = 0, total = 0;
    
    total++; pass += test_player_open();
    total++; pass += test_player_read_all_types();
    total++; pass += test_player_audio_sync();
    total++; pass += test_player_eof_handling();
    total++; pass += test_player_large_file();
    
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
