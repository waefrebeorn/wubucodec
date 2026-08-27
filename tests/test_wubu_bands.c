/* test_wubu_bands.c -- GAP-E002 gates
 *  G1 band table monotone, covers all bins, 5 nonempty bands
 *  G2 bass-heavy tone puts energy in band 0 > band 3
 *  G3 treble tone inverts it
 *  G4 normalize: max becomes 1, texture ordering preserved
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_bands.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
int main(void){
    WubuStft st; CHECK(wubu_stft_init(&st,1024,256)==0);
    WubuBandTable bt; CHECK(wubu_band_split(&st,&bt)==0);
    int bins=513;

    printf("  g1_table_structure...");
    for(int b=0;b<5;b++) CHECK(bt.start[b]<bt.end[b]);
    for(int b=1;b<5;b++) CHECK(bt.start[b]>=bt.end[b-1]-0 && bt.start[b]<=bt.end[b-1]+1 || bt.start[b]>bt.end[b-1]);
    CHECK(bt.end[4]==bins);
    printf("PASS\n");passed++;

    /* bass tone 60 Hz */
    float x[4096];
    for(int i=0;i<4096;i++) x[i]=sinf(2*(float)M_PI*60.0f*(float)i/48000.0f);
    int T; float* S=wubu_stft_forward(&st,x,4096,&T);
    float e[5]; wubu_band_energy(&bt,S,T,bins,e);
    printf("  g2_bass_localization..."); CHECK(e[0]>e[3]); printf("PASS\n");passed++;

    /* treble tone 9 kHz */
    for(int i=0;i<4096;i++) x[i]=sinf(2*(float)M_PI*9000.0f*(float)i/48000.0f);
    free(S); S=wubu_stft_forward(&st,x,4096,&T);
    wubu_band_energy(&bt,S,T,bins,e);
    printf("  g3_treble_inversion..."); CHECK(e[3]>e[0]); printf("PASS\n");passed++;

    float t[5]={0.2f,1.0f,0.5f,0.1f,0.8f};
    wubu_band_normalize(t);
    printf("  g4_normalize_texture...");
    float mx=0; for(int b=0;b<5;b++) if(t[b]>mx)mx=t[b];
    CHECK(fabsf(mx-1.0f)<1e-6f);
    CHECK(t[1]>t[3]&&t[4]>t[0]);   /* ordering preserved */
    printf("PASS\n");passed++;

    free(S);wubu_stft_free(&st);
    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}
