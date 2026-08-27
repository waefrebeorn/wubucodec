/*
 * wubu_beam.h -- Beam-sweep canvas: strip representation + band registry
 *
 * GAP-A001 (beam strip), GAP-A005 (orientation flag),
 * GAP-B003 (band registry) CLOSED here.
 */

#ifndef WUBU_BEAM_H
#define WUBU_BEAM_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    WUBU_BAND_VISIBLE  = 0,  /* rendered pixels ("light")              */
    WUBU_BAND_INFRARED = 1,  /* audio + P-frame payloads, never rendered */
    WUBU_BAND_UV       = 2,  /* extension: provenance/watermark         */
    WUBU_BAND_COUNT
} WubuBandId;

typedef struct {
    int width;      /* band width in beam-local columns            */
    bool visible;   /* false => renderer ignores this band          */
} WubuBandSpec;

typedef enum {
    WUBU_BEAM_HORIZONTAL = 0,
    WUBU_BEAM_VERTICAL   = 1
} WubuBeamOrientation;             /* GAP-A005 */

typedef enum {
    WUBU_LAYOUT_CONTIGUOUS=0,
    WUBU_LAYOUT_INTERLEAVED=1
} WubuBandLayout;
typedef struct {
    int strip_width;               /* short-axis size (e.g. 4000)  */
    int sweep_length;              /* long-axis samples per frame  */
    WubuBeamOrientation orientation;
    WubuBandSpec bands[WUBU_BAND_COUNT];
} WubuBeamConfig;

typedef struct {
    WubuBeamConfig cfg;
    float* strip;                  /* [total_width][strip_width][3] current strip */
    int total_width;               /* sum of band widths           */
    int sweep_pos;                 /* current sweep position       */
    WubuBandLayout layout;         /* GAP-B009 */
} WubuBeam;

int  wubu_beam_init(WubuBeam* beam, const WubuBeamConfig* cfg);
void wubu_beam_free(WubuBeam* beam);

int  wubu_beam_band_offset(const WubuBeam* beam, WubuBandId id);
bool wubu_beam_band_visible(const WubuBeam* beam, WubuBandId id);
float* wubu_beam_at(WubuBeam* beam, WubuBandId id, int s, int bx, int y);

void wubu_beam_write_visible(WubuBeam* beam,int s,int bx,int y,float r,float g,float b);
void wubu_beam_write_infrared(WubuBeam* beam,int s,int bx,int y,float v0,float v1,float v2);

/* renderer view of the current strip — visible bands only */
void wubu_beam_render_strip(const WubuBeam* beam,float* out);

/* GAP-B010: rate accounting — bits consumed by each band at a given
 * quantization level. bits_band = width*strip_width*log2(levels)*3 (RGB).
 * Returns total across all bands; per-band written to out[5]. */
float wubu_beam_rate_account(const WubuBeam* beam,int quant_levels,
                             float* out_bits_per_band /*[WUBU_BAND_COUNT]*/);

/* GAP-B009: interleaving policy — layout of bands along the strip.
 * CONTIGUOUS = bands stacked side by side (current buffer).
 * INTERLEAVED = columns round-robin across bands (temporal spread for
 * error resilience). Policy is metadata; wubu_beam_at resolves it. */
void wubu_beam_set_layout(WubuBeam* beam,WubuBandLayout layout);
int  wubu_beam_column_to_band(const WubuBeam* beam,int strip_col);

/* GAP-A013: beam-to-latent-field mapping — resolve a sweep position to
 * the latent-field cell it addresses, given the field grid dims. This is
 * the contract between the beam (1D stream) and the coordinate-addressable
 * canvas (wubu_canvas): any sweep sample maps deterministically onto a
 * field cell, so the encoder/decoder agree without side channel. */
typedef struct {
    int frame_index;   /* which keyframe window this sample belongs to */
    float u,v;         /* normalized [0,1) coords inside the frame */
} WubuBeamFieldCoord;
void wubu_beam_field_coord(const WubuBeam* beam,int sweep_sample,
                           int frames_per_sweep,WubuBeamFieldCoord* out);

/* GAP-B004: renderer ignore-mask — 1 per strip column if that column is
 * visible-band, 0 if invisible (renderer must skip). */
void wubu_beam_visibility_mask(const WubuBeam* beam,char* mask /*[total_width]*/);

/* GAP-B005: codec-side invisible-band reader — copies ALL invisible band
 * payloads of the current strip into out [invisible_width*strip_width*3]. */
int wubu_beam_read_invisible(const WubuBeam* beam,float* out,size_t out_cap);

/* φ-fractal visit order along the sweep (GAP-A009 foundation):
 * order[k] = sweep position of the k-th sample. */
void wubu_beam_phi_order(int* order, int sweep_length);

/* GAP-G001: decode-at-N — resolution IS sampling depth. Given a full φ
 * order over the sweep, produce the first N ranks' positions resampled to
 * ANY output resolution N_out. Decode 1080p, 4K, 8K or arbitrary N from
 * the SAME latent field by choosing depth; no re-encode. Returns 0. */
void wubu_beam_decode_at(const int* order, int sweep_length,
                         int* out_positions, int n_out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_BEAM_H */
