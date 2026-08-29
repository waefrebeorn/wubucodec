/* wubuv.h — WUBV Container Format Library
 *
 * A proper, usable video container with:
 * - Video: WuBuCodec RDO encoder (I/P frames, 8×8 blocks)
 * - Audio: Kodak STFT encoder (512-pt FFT, 128 hop)
 * - Proper A/V interleaving
 * - CRC32 integrity checking
 * - Seek index for fast navigation
 *
 * FILE LAYOUT:
 *   Offset  Size  Field
 *   ------  ----  -----
 *   0       4     magic "WUBV"
 *   4       2     version (3)
 *   6       2     flags (bit0: has_audio, bit1: has_index)
 *   8       2     width
 *   10      2     height
 *   12      2     fps
 *   14      2     frame_count
 *   16      4     audio_rate
 *   20      4     audio_channels
 *   24      4     total_audio_samples
 *   28      4     index_offset (0 if no index)
 *   32      ...   frames
 *   end-4   4     CRC32
 *
 *   FRAME:
 *   +0      1     type: 0=KEY, 1=INTER, 2=AUDIO, 3=INDEX
 *   +1      4     payload_size
 *   +5      ...   payload
 *
 *   VIDEO PAYLOAD (KEY/INTER):
 *   +0      2     original_width
 *   +2      2     original_height
 *   +4      ...   Y plane (width*height bytes)
 *
 *   AUDIO PAYLOAD:
 *   +0      2     num_samples
 *   +2      ...   float32 PCM samples
 */
#ifndef WUBUV_H
#define WUBUV_H
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBUV_VERSION 3
#define WUBUV_MAGIC "WUBV"

#define WUBUV_FLAG_HAS_AUDIO  0x01
#define WUBUV_FLAG_HAS_INDEX  0x02

#define WUBUV_FRAME_KEY   0
#define WUBUV_FRAME_INTER 1
#define WUBUV_FRAME_AUDIO 2
#define WUBUV_FRAME_INDEX 3

/* Header */
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
    uint16_t audio_channels;
    uint32_t total_audio_samples;
    uint32_t index_offset;
} WubuvHeader;
#pragma pack(pop)

/* Seek index entry */
typedef struct {
    uint32_t frame_number;
    uint32_t file_offset;
    uint8_t  frame_type;
} WubuvIndexEntry;

/* Writer */
typedef struct {
    FILE* f;
    WubuvHeader hdr;
    WubuvIndexEntry* index;
    uint32_t index_count;
    uint32_t index_capacity;
    long* frame_offsets;  /* Track frame positions for index */
    uint32_t frame_offsets_count;
} WubuvWriter;

/* Reader */
typedef struct {
    FILE* f;
    WubuvHeader hdr;
    WubuvIndexEntry* index;
    uint32_t index_size;
    uint32_t current_frame;
} WubuvReader;

/* Writer API */
WubuvWriter* wubuv_writer_open(const char* path, uint16_t width, uint16_t height,
                                uint16_t fps, uint16_t frame_count,
                                int has_audio, uint32_t audio_rate,
                                uint16_t audio_channels);
int wubuv_write_video(WubuvWriter* w, const uint8_t* compressed_data, uint32_t compressed_size, int is_key);
int wubuv_write_audio(WubuvWriter* w, const float* samples, uint16_t num_samples);
int wubuv_writer_close(WubuvWriter* w);

/* Reader API */
WubuvReader* wubuv_reader_open(const char* path);
int wubuv_read_frame(WubuvReader* r, uint8_t* type_out, uint8_t** data_out,
                      uint32_t* size_out);
int wubuv_seek_to_frame(WubuvReader* r, uint32_t frame_number);
void wubuv_reader_close(WubuvReader* r);

/* Utility */
int wubuv_verify(const char* path);
uint32_t wubuv_crc32(const uint8_t* data, size_t len);
void wubuv_print_info(const char* path);

#ifdef __cplusplus
}
#endif

#endif /* WUBUV_H */
