/* wubu_encoder.c — C15 keystone: unified RDO encoder loop
 *
 * Wires together all gated C15 components into a single-pass encoder.
 *
 * Architecture (per 8×8 block, in coding order):
 *  1. If I-frame: try 4 intra modes (DC/Vert/Horiz/Plane), pick best SAD.
 *     Reconstruct = prediction + DCT(recon - pred)/quantize/dequantize.
 *  2. If P-frame: get inter prediction (from ref frame or motion), compute
 *     residual = orig - pred, transform+quantize, reconstruct = pred + residual.
 *  3. If B-frame: bi-pred from past+future refs, same residual path.
 *  4. RD cost = SSE + λ·bits for the chosen mode.
 *  5. Write reconstructed pixels to output buffer.
 *
 * Operates on luminance plane only (Y, 1 byte/pixel) — standard video codec
 * approach. Intra prediction adapted to single-plane neighbors.
 */

#include "wubu_encoder.h"
#include "wubu_rdo.h"
#include "wubu_rdomode.h"
#include "wubu_fastmode.h"
#include "wubu_intra.h"
#include "wubu_transform.h"
#include "wubu_bframe2.h"
#include "wubu_motionest.h"
#include "wubu_deblock.h"
#include "wubu_sao.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979f
#endif

#define BS 8  /* block size */

/* ===== Local intra prediction (single-plane, Y-only) ===== */

static int has_left_y(const uint8_t* img,int W,int H,int bx,int by){
    return bx>0;
}
static int has_top_y(const uint8_t* img,int W,int H,int bx,int by){
    return by>0;
}

/* DC: average of available border pixels */
static void ip_dc_y(const uint8_t* img,int W,int H,int bx,int by,uint8_t* blk){
    int sum=0,cnt=0;
    if(bx>0) for(int r=0;r<BS;r++){int y=by*BS+r; if(y<H){sum+=img[y*W+bx-1];cnt++;}}
    if(by>0) for(int c=0;c<BS;c++){int x=bx*BS+c; if(x<W){sum+=img[(by-1)*BS*8+c];cnt++;}}
    uint8_t dc=cnt>0?(uint8_t)(sum/cnt):128;
    for(int i=0;i<64;i++)blk[i]=dc;
}

/* Vertical: replicate top row downward */
static void ip_vert_y(const uint8_t* img,int W,int H,int bx,int by,uint8_t* blk){
    if(by==0){ip_dc_y(img,W,H,bx,by,blk);return;}
    for(int c=0;c<BS;c++){int x=bx*BS+c; if(x>=W)break;
        uint8_t v=img[(by-1)*BS*8+c];
        for(int r=0;r<BS;r++)blk[r*BS+c]=v;
    }
}

/* Horizontal: replicate left column rightward */
static void ip_horiz_y(const uint8_t* img,int W,int H,int bx,int by,uint8_t* blk){
    if(bx==0){ip_dc_y(img,W,H,bx,by,blk);return;}
    for(int r=0;r<BS;r++){int y=by*BS+r; if(y>=H)break;
        uint8_t v=img[y*W+bx-1];
        for(int c=0;c<BS;c++)blk[r*BS+c]=v;
    }
}

/* Plane: weighted extrapolation from top-left corner */
static void ip_plane_y(const uint8_t* img,int W,int H,int bx,int by,uint8_t* blk){
    if(bx==0||by==0){ip_vert_y(img,W,H,bx,by,blk);return;}
    uint8_t corner=img[(by-1)*BS*8+bx-1];
    int gh=0,gv=0,n=0;
    for(int c=0;c<BS&&bx-1+c<W;c++){gh+=img[(by-1)*BS*8+bx-1+c]-corner;n++;}
    for(int r=0;r<BS&&by-1+r<H;r++){gv+=img[(by-1+r)*BS*8+bx-1]-corner;n++;}
    if(n>0){gh/=n;gv/=n;}
    for(int r=0;r<BS;r++)for(int c=0;c<BS;c++){
        int v=corner+gh*(c+1)/BS+gv*(r+1)/BS;
        blk[r*BS+c]=(uint8_t)(v<0?0:(v>255?255:v));
    }
}

/* Predict a single 8×8 block with intra mode (0=DC,1=Vert,2=Horiz,3=Plane) */
static void intra_predict_y(const uint8_t* img,int W,int H,int bx,int by,
                             int mode,uint8_t* blk){
    switch(mode){
        case 0: ip_dc_y(img,W,H,bx,by,blk);break;
        case 1: ip_vert_y(img,W,H,bx,by,blk);break;
        case 2: ip_horiz_y(img,W,H,bx,by,blk);break;
        case 3: ip_plane_y(img,W,H,bx,by,blk);break;
        default: ip_dc_y(img,W,H,bx,by,blk);
    }
}

/* Best intra mode by SAD against original block */
static int best_intra_mode(const uint8_t* img,const uint8_t* orig,
                            int W,int H,int bx,int by){
    uint8_t pred[64];
    int best=0; long best_sad=1LL<<40;
    for(int m=0;m<4;m++){
        intra_predict_y(img,W,H,bx,by,m,pred);
        long sad=0;
        for(int i=0;i<64;i++) sad+=abs((int)orig[i]-(int)pred[i]);
        if(sad<best_sad){best_sad=sad;best=m;}
    }
    return best;
}

/* ===== Inter prediction ===== */

/* For P-frame: get prediction from reference frame at block position */
static void inter_pred_plane(const uint8_t* ref,int W,int H,
                              int bx,int by,uint8_t* pred){
    int rx=bx*BS, ry=by*BS;
    for(int r=0;r<BS;r++){
        for(int c=0;c<BS;c++){
            int srcY=ry+r, srcX=rx+c;
            pred[r*BS+c]=(srcY<H&&srcX<W)?ref[srcY*W+srcX]:0;
        }
    }
}

/* Motion-compensated prediction with half-pel refinement */
static void mc_pred(const uint8_t* ref,int W,int H,
                     int mvx,int mvy,int bx,int by,uint8_t* pred){
    /* Simple half-pel MC: bilinear interpolation */
    float fx=bx*BS+mvx, fy=by*BS+mvy;
    int x0=(int)floorf(fx), y0=(int)floorf(fy);
    float dx=fx-x0, dy=fy-y0;
    for(int r=0;r<BS;r++){
        for(int c=0;c<BS;c++){
            int sy=y0+r, sx=x0+c;
            if(sy<0||sy>=H||sx<0||sx>=W){pred[r*BS+c]=0;continue;}
            float v00=ref[sy*W+sx];
            float v10=sx+1<W?ref[sy*W+sx+1]:v00;
            float v01=sy+1<H?ref[(sy+1)*W+sx]:v00;
            float v11=sx+1<W&&sy+1<H?ref[(sy+1)*W+sx+1]:v00;
            float v=v00*(1-dx)*(1-dy)+v10*dx*(1-dy)+v01*(1-dx)*dy+v11*dx*dy;
            pred[r*BS+c]=(uint8_t)(v<0?0:(v>255?255:(int)(v+0.5)));
        }
    }
}

/* ===== Bit cost estimation ===== */

static int est_block_bits(const int16_t* quant){
    int bits=0;
    for(int i=0;i<64;i++){
        if(quant[i]!=0){
            /* Matches exp-Golomb cost: code=(s>0?2s-1:-2s), bits=2*floor(log2(code+1))+1 */
            int s = quant[i];
            unsigned code = (s > 0) ? (unsigned)(2*s - 1) : (unsigned)(-2*s);
            int k = 0;
            unsigned v1 = code + 1;
            while ((v1 >> (k + 1)) && k < 30) k++;
            bits += 2*k + 1;
        }
    }
    return bits+4; /* ~4 byte flush overhead */
}

/* ===== Main encoder ===== */

int wubu_encode_frame(const uint8_t* orig,
                      const uint8_t* ref_past,
                      const uint8_t* ref_future,
                      int W,int H,int qp,
                      wubu_frame_type_t frame_type,
                      const int16_t* mv_grid,
                      uint8_t* recon_out,
                      long* total_bits){
    if(!orig||!recon_out||!total_bits) return -1;
    if(W<=0||H<=0) return -1;

    memset(recon_out,0,(size_t)W*H);
    *total_bits=0;
    double lambda=wubu_lambda_from_qp(qp,frame_type);

    /* Compute motion vectors if not provided (P/B frames) */
    int16_t* computed_mv = NULL;
    if(mv_grid == NULL && ref_past != NULL && frame_type != WUBU_I_FRAME) {
        int nbx = W / BS, nby = H / BS;
        /* wubu_me_frame writes int (4 bytes), so use int buffer */
        int* mv_buf = malloc((size_t)nbx * nby * 2 * sizeof(int));
        if(mv_buf) {
            wubu_me_frame(orig, ref_past, W, H, BS, 8, mv_buf);
            /* Convert to int16_t */
            computed_mv = malloc((size_t)nbx * nby * 2 * sizeof(int16_t));
            if(computed_mv) {
                for(int i=0;i<nbx*nby*2;i++) computed_mv[i] = (int16_t)mv_buf[i];
            }
            free(mv_buf);
        }
    }
    const int16_t* mvs = mv_grid ? mv_grid : computed_mv;

    /* encode in 8×8 blocks */
    for(int by=0;by<H;by+=BS){
        for(int bx=0;bx<W;bx+=BS){
            int npw=BS, nph=BS;
            if(bx+BS>W) npw=W-bx;
            if(by+BS>H) nph=H-by;
            if(npw<=0||nph<=0) continue;

            /* ---- I-frame: intra prediction + DCT residual ---- */
            if(frame_type==WUBU_I_FRAME){
                /* extract actual block from original */
                uint8_t org_blk[64]={0};
                for(int r=0;r<nph;r++)
                    for(int c=0;c<npw;c++)
                        org_blk[r*BS+c]=orig[(by+r)*W+(bx+c)];

                /* best intra mode */
                int mode=best_intra_mode(recon_out,org_blk,W,H,bx,by);
                uint8_t pred[64]={0};
                intra_predict_y(recon_out,W,H,bx,by,mode,pred);

                /* residual */
                int16_t res[64]={0};
                for(int i=0;i<npw*nph;i++)
                    res[i]=(int16_t)((int)org_blk[i]-(int)pred[i]);

                /* transform + quantize using simple shift (matches integrated codec) */
                int16_t coeffs[64]={0};
                wubu_tr_forward(res,coeffs,BS);
                int qshift = qp / 2;
                if (qshift < 1) qshift = 1;
                int qstep = qshift + 1;
                int16_t trellis_levels[64];
                for(int i=0;i<64;i++) trellis_levels[i] = (int16_t)(coeffs[i] / qstep);

                /* Use simple rounded levels for reconstruction */
                int16_t dq[64], recon_blk[64];
                for(int i=0;i<64;i++) dq[i] = (int16_t)(trellis_levels[i] * qstep);
                wubu_tr_inverse(dq, recon_blk, BS);

                /* bits estimate from levels (exp-Golomb cost) */
                int bits=0;
                for(int i=0;i<64;i++)
                    if(trellis_levels[i]!=0) bits+=tr_bits_for_level(trellis_levels[i]);
                *total_bits+=bits+4;

                /* reconstruct: inverse DCT + add pred */
                for(int r=0;r<nph;r++)
                    for(int c=0;c<npw;c++){
                        int val=recon_blk[r*BS+c]+pred[r*BS+c];
                        if(val<0)val=0; if(val>255)val=255;
                        recon_out[(by+r)*W+(bx+c)]=(uint8_t)val;
                    }
                continue;
            }

            /* ---- P/B-frame: inter prediction + residual ---- */
            uint8_t pred[64]={0};

            if(frame_type==WUBU_B_FRAME&&ref_past&&ref_future){
                /* bi-pred: average past + future */
                uint8_t p_past[64]={0}, p_future[64]={0};
                inter_pred_plane(ref_past,W,H,bx,by,p_past);
                inter_pred_plane(ref_future,W,H,bx,by,p_future);
                for(int i=0;i<npw*nph;i++)
                    pred[i]=(uint8_t)((p_past[i]+p_future[i]+1)>>1);
            }else if(ref_past){
                /* P-frame: MC from past ref with MV */
                int mvx=0,mvy=0;
                if(mvs){
                    int idx=by/BS*(W/BS)+bx/BS;
                    if(idx>=0&&idx<(W/BS)*(H/BS)){
                        mvx=mvs[idx*2];
                        mvy=mvs[idx*2+1];
                    }
                }
                mc_pred(ref_past,W,H,mvx,mvy,bx,by,pred);
            }else{
                /* fallback: no reference */
                inter_pred_plane(NULL,W,H,bx,by,pred);
            }

            /* extract actual block from original */
            uint8_t org_blk[64]={0};
            for(int r=0;r<nph;r++)
                for(int c=0;c<npw;c++)
                    org_blk[r*BS+c]=orig[(by+r)*W+(bx+c)];

            /* residual = orig - pred */
            int16_t res[64]={0};
            for(int i=0;i<npw*nph;i++)
                res[i]=(int16_t)((int)org_blk[i]-(int)pred[i]);

            /* ===== C9: variable-size transform selection ===== */
            /* Try candidate transforms, pick best RD (SSE + λ·bits) */
            int qshift = qp / 2;

            /* candidate 1: 8x8 DCT (baseline) */
            int16_t coeffs8[64]={0};
            wubu_tr_forward(res,coeffs8,BS);
            double dcoeffs8[64];
            for(int i=0;i<64;i++)dcoeffs8[i]=(double)coeffs8[i];
            int16_t levels8[64];
            double qstep8 = 1 << qshift;
            wubu_trellis_quantize(dcoeffs8,64,qstep8,lambda,levels8);
            int16_t dq8[64], recon8[64];
            for(int i=0;i<64;i++) dq8[i]=(int16_t)(levels8[i]*qstep8);
            wubu_tr_inverse(dq8,recon8,BS);
            int bits8=0;
            for(int i=0;i<64;i++) if(levels8[i]!=0) bits8+=tr_bits_for_level(levels8[i]);
            long sse8=0;
            for(int i=0;i<npw*nph;i++){
                int val=pred[i]+recon8[i];
                if(val<0)val=0; if(val>255)val=255;
                int d=org_blk[i]-val; sse8+=d*d;
                }
            double rd8=(double)sse8+lambda*bits8;

            /* candidate 2: 8x8 DST-VII (intra only) */
            int16_t coeffs_dst7[64]={0};
            double rd_dst7=1e30;
            int16_t levels_dst7[64], dq_dst7[64], recon_dst7[64];
            if(frame_type==WUBU_I_FRAME){
                wubu_tr_forward_dst7_4x4(res,coeffs_dst7);
                double dcoeffs_dst7[64];
                for(int i=0;i<64;i++)dcoeffs_dst7[i]=(double)coeffs_dst7[i];
                double qstep_dst7 = 1 << qshift;
                wubu_trellis_quantize(dcoeffs_dst7,64,qstep_dst7,lambda,levels_dst7);
                int16_t dq_dst7[64], recon_dst7[64];
                for(int i=0;i<64;i++) dq_dst7[i]=(int16_t)(levels_dst7[i]*qstep_dst7);
                wubu_tr_inverse_dst7_4x4(dq_dst7,recon_dst7);
                int bits_dst7=0;
                for(int i=0;i<64;i++) if(levels_dst7[i]!=0) bits_dst7+=tr_bits_for_level(levels_dst7[i]);
                long sse_dst7=0;
                for(int i=0;i<npw*nph;i++){
                    int val=pred[i]+recon_dst7[i];
                    if(val<0)val=0; if(val>255)val=255;
                    int d=org_blk[i]-val; sse_dst7+=d*d;
                }
                rd_dst7=(double)sse_dst7+lambda*bits_dst7;
            }

            /* candidate 3: 4x4 DCT (tile 8x8 block into four 4x4 sub-blocks) */
            int16_t coeffs4[64]={0};
            int16_t levels4[64], dq4[64], recon4[64]={0};
            double rd4=1e30;
            {
                int16_t sub_in[4][16], sub_out[4][16];
                for(int sb=0;sb<4;sb++){
                    int sy=(sb/2)*4, sx=(sb%2)*4;
                    for(int r=0;r<4;r++)
                        for(int c=0;c<4;c++)
                            sub_in[sb][r*4+c]=(int16_t)res[sy*BS+sx+c];
                    wubu_tr_forward_size(4,sub_in[sb],sub_out[sb]);
                }
                for(int sb=0;sb<4;sb++)
                    for(int i=0;i<16;i++)
                        coeffs4[sb*16+i]=sub_out[sb][i];
                double dcoeffs4[64];
                for(int i=0;i<64;i++)dcoeffs4[i]=(double)coeffs4[i];
                double qstep4 = 1 << qshift;
                wubu_trellis_quantize(dcoeffs4,64,qstep4,lambda,levels4);
                int16_t dq4[64];
                for(int i=0;i<64;i++) dq4[i]=(int16_t)(levels4[i]*qstep4);
                for(int sb=0;sb<4;sb++){
                    int16_t sub_dq[16], sub_recon[16];
                    for(int i=0;i<16;i++) sub_dq[i]=dq4[sb*16+i];
                    wubu_tr_inverse_size(4,sub_dq,sub_recon);
                    int sy=(sb/2)*4, sx=(sb%2)*4;
                    for(int r=0;r<4;r++)
                        for(int c=0;c<4;c++)
                            recon4[sy*BS+sx+c]=sub_recon[r*4+c];
                }
                int bits4=0;
                for(int i=0;i<64;i++) if(levels4[i]!=0) bits4+=tr_bits_for_level(levels4[i]);
                long sse4=0;
                for(int i=0;i<npw*nph;i++){
                    int val=pred[i]+recon4[i];
                    if(val<0)val=0; if(val>255)val=255;
                    int d=org_blk[i]-val; sse4+=d*d;
                }
                rd4=(double)sse4+lambda*bits4;
            }

            /* pick best transform: lowest RD cost */
            int best_tr=0; /* 0=8x8 DCT, 1=DST-VII (intra only), 2=4x4 DCT */
            double rd_best=rd8;
            int16_t* best_levels=levels8;
            int16_t* best_recon=recon8;

            if(rd_dst7<rd_best && frame_type==WUBU_I_FRAME){
                rd_best=rd_dst7; best_tr=1; best_levels=levels_dst7;
                best_recon=recon_dst7;
            }
            if(rd4<rd_best){
                rd_best=rd4; best_tr=2; best_levels=levels4;
                best_recon=recon4;
            }

            /* use best transform for reconstruction */
            int bits=0;
            for(int i=0;i<64;i++) if(best_levels[i]!=0) bits+=tr_bits_for_level(best_levels[i]);

            /* compute SSE for both paths */
            /* SKIP: SSE(orig, pred) */
            long sse_skip=0;
            for(int i=0;i<npw*nph;i++){
                int d=org_blk[i]-pred[i];
                sse_skip+=d*d;
            }
            double rd_skip=(double)sse_skip+lambda*1; /* ~1 bit flag */

            /* RESIDUAL: SSE(orig, pred + recon) */
            long sse_res=0;
            for(int i=0;i<npw*nph;i++){
                int val=pred[i]+best_recon[i];
                if(val<0)val=0; if(val>255)val=255;
                int d=org_blk[i]-val;
                sse_res+=d*d;
            }
            double rd_res=(double)sse_res+lambda*bits;

            if(rd_skip<rd_res){
                /* SKIP mode: use prediction directly, no residual */
                for(int r=0;r<nph;r++)
                    for(int c=0;c<npw;c++)
                        recon_out[(by+r)*W+(bx+c)]=pred[r*BS+c];
            }else{
                /* RESIDUAL mode: write pred + best recon */
                *total_bits+=bits;
                for(int r=0;r<nph;r++)
                    for(int c=0;c<npw;c++){
                        int val=pred[r*BS+c]+best_recon[r*BS+c];
                        if(val<0)val=0; if(val>255)val=255;
                        recon_out[(by+r)*W+(bx+c)]=(uint8_t)val;
                    }
            }
        }
    }

    /* add CABAC stream overhead: header + end-of-stream */
    *total_bits+=8; /* rough header + EOS */

    /* C12: deblocking filter on reconstructed frame (single-plane) */
    wubu_db_filter_plane(recon_out, W, H, qp);

    /* C13: SAO (Sample Adaptive Offset) post-filter */
    /* Estimate and apply band offset */
    {
        int band_offsets[4] = {0};
        wubu_sao_band_estimate(orig, recon_out, W, H, 0, band_offsets, 4);
        wubu_sao_band(recon_out, recon_out, W, H, 0, band_offsets, 4);
        
        /* Estimate and apply edge offset */
        for (int dir = 0; dir < 4; dir++) {
            int eo_offsets[5] = {0};
            wubu_sao_edge_estimate(orig, recon_out, W, H, dir, eo_offsets);
            wubu_sao_edge(recon_out, recon_out, W, H, dir, eo_offsets);
        }
    }

    if(computed_mv) free(computed_mv);
    return 0;
}

/* ===== RD curve ===== */

int wubu_rd_curve(const uint8_t* orig,
                  const uint8_t* ref_past,
                  const uint8_t* ref_future,
                  int W,int H,
                  const int* qp_vals,int n_qp,
                  wubu_frame_type_t frame_type,
                  const int16_t* mv_grid,
                  RdCurvePoint* curve){
    if(!orig||!curve) return -1;
    for(int i=0;i<n_qp;i++){
        uint8_t* recon=calloc((size_t)W*H,1);
        if(!recon) return -1;
        long bits=0;
        if(wubu_encode_frame(orig,ref_past,ref_future,W,H,
                             qp_vals[i],frame_type,mv_grid,
                             recon,&bits)!=0){
            free(recon); return -1;
        }
        long mse=0;
        for(long p=0;p<(long)W*H;p++){
            int d=orig[p]-recon[p]; mse+=d*d;
        }
        mse/=(double)W*H;
        double psnr=mse>0?10.0*log10(255.0*255.0/mse):99.9;
        curve[i].qp=qp_vals[i];
        curve[i].bits=bits;
        curve[i].psnr=psnr;
        free(recon);
    }
    return 0;
}
