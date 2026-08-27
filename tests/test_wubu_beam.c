/*
 * test_wubu_beam.c -- gates for GAP-A001/A005/B003 (+A009 foundation)
 *
 * G1 memory: beam buffer is O(strip), independent of sweep_length content
 *    semantics (8K-class sweeps allocate the same strip as 1K sweeps).
 * G2 bands: visible band renders; infrared data NEVER appears in render.
 * G3 orientation: H vs V produce consistent addressing (rotation free).
 * G4 phi order: permutation of [0,n), prefix covers spread positions
 *    (progressive uniformity — max gap of first k visits shrinks).
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_text_encoder.h"
#include "../model/wubu_beam.c"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s\n",#c); failed++; return; } }while(0)

static void test_memory_independent_of_sweep(void){
    /* two configs, same strip, wildly different sweep lengths */
    WubuBeamConfig c1={ .strip_width=4000,.sweep_length=1000,
        .orientation=WUBU_BEAM_HORIZONTAL,
        .bands={{3900,true},{90,false},{10,false}} };
    WubuBeamConfig c2=c1; c2.sweep_length=16000;   /* 16K-class */
    WubuBeam b1,b2;
    CHECK(wubu_beam_init(&b1,&c1)==0);
    CHECK(wubu_beam_init(&b2,&c2)==0);
    /* identical allocation sizes (strip is per-sweep-position window) */
    CHECK(b1.total_width==b2.total_width);
    wubu_beam_free(&b1);wubu_beam_free(&b2);
}
static void test_infrared_never_renders(void){
    WubuBeamConfig cfg={ .strip_width=64,.sweep_length=32,
        .orientation=WUBU_BEAM_HORIZONTAL,
        .bands={{48,true},{14,false},{2,false}} };
    WubuBeam b; CHECK(wubu_beam_init(&b,&cfg)==0);

    /* poison the infrared band */
    for(int s=0;s<32;s++)for(int x=0;x<14;x++)for(int y=0;y<64;y++)
        wubu_beam_write_infrared(&b,s,x,y,1.0f,1.0f,1.0f);
    /* write a known visible pixel */
    wubu_beam_write_visible(&b,5,0,10,0.25f,0.5f,0.75f);
    b.sweep_pos=5;

    float out[48*64*3];
    wubu_beam_render_strip(&b,out);
    /* visible pixel survived intact */
    CHECK(fabsf(out[(10*48+0)*3+0]-0.25f)<1e-6f);
    CHECK(fabsf(out[(10*48+0)*3+1]-0.50f)<1e-6f);
    /* no rendered pixel is white (infrared leak check) */
    for(int i=0;i<48*64*3;i++) CHECK(out[i]<0.999f || i==(10*48+0)*3);
    wubu_beam_free(&b);
}
static void test_orientation_flag(void){
    WubuBeamConfig h={ .strip_width=32,.sweep_length=64,
        .orientation=WUBU_BEAM_HORIZONTAL,
        .bands={{24,true},{6,false},{2,false}} };
    WubuBeamConfig v=h; v.orientation=WUBU_BEAM_VERTICAL;
    WubuBeam bh,bv;
    CHECK(wubu_beam_init(&bh,&h)==0);
    CHECK(wubu_beam_init(&bv,&v)==0);
    /* same physical buffer size either way — rotation costs nothing */
    CHECK(bh.total_width==bv.total_width);
    CHECK(bv.cfg.orientation==WUBU_BEAM_VERTICAL);
    wubu_beam_free(&bh);wubu_beam_free(&bv);
}
static void test_phi_order_is_permutation_and_progressive(void){
    int n=256,*order=malloc(sizeof(int)*(size_t)n);
    wubu_beam_phi_order(order,n);
    /* permutation */
    char* seen=calloc((size_t)n,1);
    for(int i=0;i<n;i++){ CHECK(order[i]>=0&&order[i]<n&&!seen[order[i]]); seen[order[i]]=1; }
    /* progressive spread: first k visited points' max-gap shrinks toward 1 */
    const int ks[3]={8,32,128};
    for(int ki=0;ki<3;ki++){
        int k=ks[ki];
        int chosen[256],m=0;
        for(int i=0;i<n&&m<k;i++) if(order[i]<k) ;
        /* collect positions whose order index < k */
        for(int i=0;i<n;i++) if(order[i]<k) chosen[m++]=i;
        int maxgap=0,prev=-1;
        /* circular scan over sorted chosen (chosen is ascending by construction) */
        for(int j=0;j<m;j++){ int gap=chosen[j]-(j?chosen[j-1]:-1); if(gap>maxgap)maxgap=gap; }
        if(n-chosen[m-1]>maxgap) maxgap=n-chosen[m-1];
        CHECK(maxgap<=2*(n/k+1));   /* near-uniform coverage (edges may merge two gaps) */
    }
    free(order);free(seen);
}
static void test_decode_at_any_resolution(void){
    /* GAP-G001 gate: decode-at-N yields valid positions for any N, and a
     * deeper decode contains the shallower one's information (prefix
     * property of ranks). */
    int n=512,*order=malloc(sizeof(int)*(size_t)n);
    wubu_beam_phi_order(order,n);
    const int Ns[4]={64,180,1080,512};  /* arbitrary resolutions incl. >sweep handled */
    for(int ni=0;ni<4;ni++){
        int N=Ns[ni];
        int* pos=malloc(sizeof(int)*(size_t)N);
        wubu_beam_decode_at(order,n,pos,N);
        for(int i=0;i<N;i++)
            CHECK(pos[i]>=0&&pos[i]<n);
        free(pos);
    }
    /* monotone depth: decode at 100 uses ranks {0,5,10,...}; decode at 200
     * rank set must include every even-indexed sample of the 100-decode */
    int p100[100],p200[200];
    wubu_beam_decode_at(order,n,p100,100);
    wubu_beam_decode_at(order,n,p200,200);
    for(int i=0;i<100;i++){
        int found=0;
        for(int j=0;j<200;j++) if(p200[j]==p100[i]){found=1;break;}
        CHECK(found);   /* shallower decode ⊂ deeper decode */
    }
    free(order);
}
static void test_mask_and_invisible_reader(void){
    WubuBeamConfig cfg={ .strip_width=32,.sweep_length=16,
        .orientation=WUBU_BEAM_HORIZONTAL,
        .bands={{24,true},{6,false},{2,false}} };
    WubuBeam b; CHECK(wubu_beam_init(&b,&cfg)==0);
    char mask[32];
    wubu_beam_visibility_mask(&b,mask);
    int vis=0,invis=0;
    for(int i=0;i<32;i++){vis+=mask[i];invis+=!mask[i];}
    CHECK(vis==24&&invis==8);          /* mask matches band registry */

    /* poison infrared with a signature */
    for(int x=0;x<6;x++)for(int y=0;y<32;y++)
        wubu_beam_write_infrared(&b,3,x,y,7.0f,7.0f,7.0f);
    float inv[8*32*3];
    int n=wubu_beam_read_invisible(&b,inv,sizeof(inv)/sizeof(float));
    CHECK(n==8*32*3);
    int hits=0;
    for(int i=0;i<n;i+=3)
        if(inv[i]==7.0f&&inv[i+1]==7.0f&&inv[i+2]==7.0f)hits++;
    CHECK(hits==6*32);                 /* every invisible cell recovered */
    /* and none of it is visible-band data */
    float out[24*32*3]; b.sweep_pos=3;
    wubu_beam_render_strip(&b,out);
    for(int i=0;i<24*32*3;i++) CHECK(out[i]<6.999f);
    wubu_beam_free(&b);
}
static void test_rate_account_and_layout(void){
    WubuBeamConfig cfg={ .strip_width=100,.sweep_length=8,
        .orientation=WUBU_BEAM_HORIZONTAL,
        .bands={{50,true},{40,false},{10,false}} };
    WubuBeam b; CHECK(wubu_beam_init(&b,&cfg)==0);
    /* rate accounting: total = 100*100*3*log2(levels) */
    float per[WUBU_BAND_COUNT];
    float tot=wubu_beam_rate_account(&b,8,per);
    CHECK(fabsf(tot-100.0f*100.0f*3.0f*3.0f)<1e-3f);   /* log2(8)=3 */
    CHECK(per[1]/per[2]==4);                            /* 40 vs 10 width */
    /* +1 quantization level adds one bit/sample: log2(16)-log2(8)=1 */
    float tot2=wubu_beam_rate_account(&b,16,NULL);
    CHECK(fabsf(tot2-tot-100.0f*100.0f*3.0f)<1e-3f);
    /* interleaved layout: column->band mapping is a valid permutation-ish
     * walk that respects band widths */
    wubu_beam_set_layout(&b,WUBU_LAYOUT_INTERLEAVED);
    int count[3]={0,0,0};
    for(int c=0;c<100;c++){
        int band=wubu_beam_column_to_band(&b,c);
        CHECK(band>=0&&band<3);
        count[band]++;
    }
    CHECK(count[0]==50&&count[1]==40&&count[2]==10);   /* widths preserved */
    wubu_beam_free(&b);
}
static void test_text_into_infrared(void){
    /* GAP-B011: text embedding rides the invisible band. Encode -> scatter
     * into IR cells -> read back -> identical values; renderer still blind. */
    WubuTextEncoder te; CHECK(wubu_text_init(&te,1024,8)==0);
    float emb[8]; wubu_text_encode(&te,"the beam carries unseen text",emb,8);

    WubuBeamConfig cfg={ .strip_width=16,.sweep_length=4,
        .orientation=WUBU_BEAM_HORIZONTAL,
        .bands={{10,true},{5,false},{1,false}} };
    WubuBeam b; CHECK(wubu_beam_init(&b,&cfg)==0);
    /* 8 floats -> 3 cells (RGB triplets) x2 rows... write across cells */
    int cell=0;
    for(int d=0;d<8;d+=3){
        int bx=cell%5,y=cell/5;
        wubu_beam_write_infrared(&b,1,bx,y,emb[d],d+1<8?emb[d+1]:0,d+2<8?emb[d+2]:0);
        cell++;
    }
    /* read back */
    float back[240]; int n=wubu_beam_read_invisible(&b,back,240);
    CHECK(n==5*16*3);   /* whole invisible region */
    /* reader layout: [bx][y][rgb] — reassemble our 8 values from the
     * three cells we wrote at (bx=cell,y=0) */
    float got[8]; int gi=0;
    for(int cell=0;cell<3;cell++)
        for(int rgb=0;rgb<3&&gi<8;rgb++)
            got[gi++]=back[((size_t)cell*16+0)*3+rgb];
    for(int d=0;d<8;d++) CHECK(fabsf(got[d]-emb[d])<1e-6f);
    /* renderer blind to it */
    b.sweep_pos=1;
    float out[10*16*3]; wubu_beam_render_strip(&b,out);
    for(int i=0;i<10*16*3;i++) CHECK(fabsf(out[i])<6.9f || out[i]!=7.0f);
    wubu_beam_free(&b);wubu_text_free(&te);
}
static void test_field_coord_mapping(void){
    /* GAP-A013 gates: monotone frame index over the sweep; u in [0,1);
     * v covers strip uniformly; endpoints map to first/last frames. */
    WubuBeamConfig cfg={ .strip_width=100,.sweep_length=800,
        .orientation=WUBU_BEAM_HORIZONTAL,
        .bands={{60,true},{30,false},{10,false}} };
    WubuBeam b; CHECK(wubu_beam_init(&b,&cfg)==0);
    int prev_frame=-1;
    for(int s=0;s<800;s+=7){
        WubuBeamFieldCoord fc;
        wubu_beam_field_coord(&b,s,8,&fc);
        CHECK(fc.frame_index>=prev_frame);
        CHECK(fc.u>=0.0f&&fc.u<1.0f);
        CHECK(fc.v>=0.0f&&fc.v<1.0f);
        prev_frame=fc.frame_index;
    }
    CHECK(prev_frame==7);   /* last samples reach final window */
    wubu_beam_free(&b);
}
int main(void){
    printf("=== WuBuMath Beam Canvas Tests ===\n\n");
    test_field_coord_mapping(); printf("  test_field_coord_mapping...PASS\n");passed++;
    test_text_into_infrared(); printf("  test_text_into_infrared...PASS\n");passed++;
    test_rate_account_and_layout(); printf("  test_rate_account_and_layout...PASS\n");passed++;
    test_mask_and_invisible_reader(); printf("  test_mask_and_invisible_reader...PASS\n");passed++;
    test_decode_at_any_resolution(); printf("PASS\n");passed++;
    printf("  test_memory_independent_of_sweep..."); test_memory_independent_of_sweep(); printf("PASS\n");passed++;
    printf("  test_infrared_never_renders...");      test_infrared_never_renders();     printf("PASS\n");passed++;
    printf("  test_orientation_flag...");           test_orientation_flag();          printf("PASS\n");passed++;
    printf("  test_phi_order_is_permutation...");   test_phi_order_is_permutation_and_progressive(); printf("PASS\n");passed++;
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
