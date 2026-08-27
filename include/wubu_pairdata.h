/* GAP-D006: deterministic synthetic pair corpus for manifold-CLIP */
#ifndef WUBU_PAIRDATA_H
#define WUBU_PAIRDATA_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int num_pairs;
    int feat_dim;
    float caption_noise;   /* channel noise between image and caption */
    float val_split;       /* fraction held out (0..0.5) */
    unsigned seed;
} WubuPairDataConfig;

typedef struct {
    WubuPairDataConfig cfg;
    float* image_feat;  /* [num_pairs, F] */
    float* text_feat;   /* [num_pairs, F] */
    int train_n;        /* first train_n = train, rest = val */
} WubuPairCorpus;

int  wubu_pairdata_init(WubuPairCorpus* c,const WubuPairDataConfig* cfg);
void wubu_pairdata_free(WubuPairCorpus* c);
#ifdef __cplusplus
}
#endif
#endif
