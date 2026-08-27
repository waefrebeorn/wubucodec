/*
 * wubu_gaad_encoder.c -- GAAD + WuBu Nested Fractal Encoder
 * Faithful C implementation of WuBuGAADHybridGen_v0.2.py
 */
#include "wubu_gaad_encoder.h"
#define PHI WUBU_PHI
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

PoincareBall poincare_ball(float c) {
    PoincareBall pb;
    pb.c = c;
    pb.c_inv = 1.0f / sqrtf(c > 1e-8f ? c : 1e-8f);
    pb.one_minus_c = 1.0f - c;
    return pb;
}
void poincare_project(float* x, int dim, PoincareBall* pb) {
    float n2 = 0; for (int i = 0; i < dim; i++) n2 += x[i]*x[i];
    float max_n2 = 0.999f / pb->c; /* ||x||^2 < 1/c */
    if (n2 > max_n2) { float s = sqrtf(max_n2 / (n2 + 1e-10f)); for (int i = 0; i < dim; i++) x[i] *= s; }
}
void poincare_expmap0(float* v, float* out, int dim, PoincareBall* pb) {
    float vn2 = 0; for (int i = 0; i < dim; i++) vn2 += v[i]*v[i];
    float vn = sqrtf(vn2);
    if (vn < 1e-8f) { for (int i = 0; i < dim; i++) out[i] = v[i]; return; }
    float arg = pb->c_inv * vn;
    if (arg > 10.0f) arg = 10.0f; /* prevent tanh saturation issues */
    float th = tanhf(arg);
    float sc = th / (pb->c_inv * vn);
    for (int i = 0; i < dim; i++) out[i] = v[i] * sc;
    poincare_project(out, dim, pb);
}
void poincare_logmap0(float* x, float* out, int dim, PoincareBall* pb) {
    float xn2 = 0; for (int i = 0; i < dim; i++) xn2 += x[i]*x[i];
    float xn = sqrtf(xn2);
    if (xn < 1e-8f) { for (int i = 0; i < dim; i++) out[i] = x[i]; return; }
    float arg = pb->c_inv * xn;
    if (arg > 0.999f) arg = 0.999f; /* prevent atanh domain error */
    float ah = atanhf(arg);
    float sc = ah / (pb->c_inv * xn);
    for (int i = 0; i < dim; i++) out[i] = x[i] * sc;
}
void poincare_mobius_add(float* x, float* y, float* out, int dim, PoincareBall* pb) {
    float x2=0,y2=0,xy=0;
    for (int i=0;i<dim;i++){x2+=x[i]*x[i];y2+=y[i]*y[i];xy+=x[i]*y[i];}
    float c=pb->c, den=1.0f+2.0f*c*xy+c*c*x2*y2;
    if(den<1e-8f)den=1e-8f;
    for(int i=0;i<dim;i++) out[i]=((1.0f+2.0f*c*xy+c*y2)*x[i]+(1.0f-c*x2)*y[i])/den;
    poincare_project(out,dim,pb);
}
int generate_phi_spiral(SpiralPoint* points, int num_points, int W, int H) {
    if(num_points<=0)return 0;
    float cx=(float)W/2.0f, cy=(float)H/2.0f;
    float a=0.05f*(float)(W<H?W:H), b=WUBU_LOG_PHI_OVER_PI_2;
    points[0].x=cx;points[0].y=cy;points[0].scale=0.25f;
    int gen=1; float ang=0.0f;
    float step=PHI*2.0f*M_PI/(float)(num_points>1?num_points-1:1);
    for(int i=0;i<num_points-1&&gen<num_points;i++){
        float r=a*expf(b*ang), mr=(float)(W>H?W:H)*0.6f;
        if(r>mr)r=mr;
        float x=cx+r*cosf(ang), y=cy+r*sinf(ang);
        if(x<0)x=0;if(x>=(float)W)x=(float)(W-1);
        if(y<0)y=0;if(y>=(float)H)y=(float)(H-1);
        points[gen].x=x;points[gen].y=y;
        points[gen].scale=0.20f*expf(-0.5f*r/((float)(W<H?W:H)*0.1f));
        if(points[gen].scale<0.05f)points[gen].scale=0.05f;
        gen++; ang+=step;
    }
    while(gen<num_points){points[gen]=points[gen-1];gen++;}
    return gen;
}
int golden_subdivide(Rect* rects, int W, int H, int num_target) {
    Rect queue[256]; int qh=0,qt=0,cnt=0;
    queue[qt++]=(Rect){0,0,(float)W,(float)H};
    while(qh<qt&&cnt<num_target*3&&qt<256){
        Rect r=queue[qh++]; float w=r.x2-r.x1,h=r.y2-r.y1;
        if(w<5||h<5)continue;
        if(w>h+0.01f){float c=w/PHI;
            if(c>=5&&h>=5&&cnt<num_target*3)rects[cnt++]=(Rect){r.x1,r.y1,r.x1+c,r.y2};
            if(c>=5&&h>=5&&qt<256)queue[qt++]=(Rect){r.x1,r.y1,r.x1+c,r.y2};
            if(w-c>=5&&h>=5&&cnt<num_target*3)rects[cnt++]=(Rect){r.x1+c,r.y1,r.x2,r.y2};
            if(w-c>=5&&h>=5&&qt<256)queue[qt++]=(Rect){r.x1+c,r.y1,r.x2,r.y2};
        }else if(h>w+0.01f){float c=h/PHI;
            if(w>=5&&c>=5&&cnt<num_target*3)rects[cnt++]=(Rect){r.x1,r.y1,r.x2,r.y1+c};
            if(w>=5&&c>=5&&qt<256)queue[qt++]=(Rect){r.x1,r.y1,r.x2,r.y1+c};
            if(w>=5&&h-c>=5&&cnt<num_target*3)rects[cnt++]=(Rect){r.x1,r.y1+c,r.x2,r.y2};
            if(w>=5&&h-c>=5&&qt<256)queue[qt++]=(Rect){r.x1,r.y1+c,r.x2,r.y2};
        }else{float c=w/PHI;
            if(c>=5&&h>=5&&cnt<num_target*3)rects[cnt++]=(Rect){r.x1,r.y1,r.x1+c,r.y2};
            if(c>=5&&h>=5&&qt<256)queue[qt++]=(Rect){r.x1,r.y1,r.x1+c,r.y2};
            if(w-c>=5&&h>=5&&cnt<num_target*3)rects[cnt++]=(Rect){r.x1+c,r.y1,r.x2,r.y2};
            if(w-c>=5&&h>=5&&qt<256)queue[qt++]=(Rect){r.x1+c,r.y1,r.x2,r.y2};
        }
    }
    return cnt<num_target?cnt:num_target;
}
static void wubu_level_init(WubuLevel* lvl, int dim, int level_idx,
                             int num_boundaries, float c_base, int flow_hidden) {
    memset(lvl,0,sizeof(WubuLevel));
    lvl->dim=dim;lvl->level_idx=level_idx;lvl->c_base=c_base;
    float pp=(float)(level_idx%4)-1.5f;
    lvl->c=c_base*powf(PHI,pp);if(lvl->c<0.01f)lvl->c=0.01f;
    lvl->num_boundaries=num_boundaries;lvl->flow_hidden=flow_hidden;
    lvl->use_rotation=1;lvl->spread=0.1f;
    int cd=dim*3+1;
    lvl->boundary_points=(float*)calloc(num_boundaries*dim,sizeof(float));
    lvl->descriptor=(float*)calloc(dim,sizeof(float));
    lvl->flow_w1=(float*)calloc(flow_hidden*cd,sizeof(float));
    lvl->flow_b1=(float*)calloc(flow_hidden,sizeof(float));
    lvl->flow_w2=(float*)calloc(dim*flow_hidden,sizeof(float));
    lvl->flow_b2=(float*)calloc(dim,sizeof(float));
    lvl->rot_param=(float*)calloc(1,sizeof(float));
    int nd=dim; lvl->trans_hidden=dim*2;
    lvl->trans_w1=(float*)calloc(lvl->trans_hidden*cd,sizeof(float));
    lvl->trans_b1=(float*)calloc(lvl->trans_hidden,sizeof(float));
    lvl->trans_w2=(float*)calloc(nd*lvl->trans_hidden,sizeof(float));
    lvl->trans_b2=(float*)calloc(nd,sizeof(float));
    unsigned int r=42+level_idx*17;
    for(int i=0;i<num_boundaries*dim;i++){r=r*1103515245u+12345u;lvl->boundary_points[i]=((float)(r&0x7FFF)/32768.0f-0.5f)*0.02f;}
    for(int i=0;i<dim;i++){r=r*1103515245u+12345u;lvl->descriptor[i]=((float)(r&0x7FFF)/32768.0f-0.5f)*0.02f;}
    float l1=sqrtf(6.0f/(cd+flow_hidden));
    for(int i=0;i<flow_hidden*cd;i++){r=r*1103515245u+12345u;lvl->flow_w1[i]=((float)(r&0x7FFF)/32768.0f*2.0f-1.0f)*l1;}
    lvl->rot_param[0]=0.0f;
    float l2=sqrtf(6.0f/(cd+lvl->trans_hidden));
    for(int i=0;i<lvl->trans_hidden*cd;i++){r=r*1103515245u+12345u;lvl->trans_w1[i]=((float)(r&0x7FFF)/32768.0f*2.0f-1.0f)*l2;}
    float l3=sqrtf(6.0f/(lvl->trans_hidden+nd));
    for(int i=0;i<nd*lvl->trans_hidden;i++){r=r*1103515245u+12345u;lvl->trans_w2[i]=((float)(r&0x7FFF)/32768.0f*2.0f-1.0f)*l3;}
}
static void wubu_level_forward(WubuLevel* lvl, const float* v_in, float* v_out, float* ws) {
    int dim=lvl->dim, nb=lvl->num_boundaries, fh=lvl->flow_hidden;
    PoincareBall pb=poincare_ball(lvl->c);
    float tv[64]; poincare_logmap0((float*)v_in,tv,dim,&pb);
    float bm[64]; memset(bm,0,dim*sizeof(float));
    for(int p=0;p<nb;p++)for(int d=0;d<dim;d++)bm[d]+=lvl->boundary_points[p*dim+d];
    for(int d=0;d<dim;d++)bm[d]/=(float)nb;
    float* ctx=ws; int cd=dim*3+1, ci=0;
    for(int d=0;d<dim;d++)ctx[ci++]=tv[d];
    for(int d=0;d<dim;d++)ctx[ci++]=bm[d];
    for(int d=0;d<dim;d++)ctx[ci++]=lvl->descriptor[d];
    ctx[ci++]=lvl->spread;
    float* hb=ws+cd, *hba=hb+fh, *fv=hba+fh;
    for(int h=0;h<fh;h++){float s=lvl->flow_b1[h];for(int c=0;c<cd;c++)s+=lvl->flow_w1[h*cd+c]*ctx[c];hb[h]=s;hba[h]=gelu(s);}
    for(int o=0;o<dim;o++){float s=lvl->flow_b2[o];for(int h=0;h<fh;h++)s+=lvl->flow_w2[o*fh+h]*hba[h];fv[o]=s;}
    float to[64]; for(int i=0;i<dim;i++)to[i]=tv[i]+fv[i];
    float ns=0; for(int i=0;i<dim;i++)ns+=to[i]*to[i];
    if(ns>100.0f){float sc=10.0f/sqrtf(ns);for(int i=0;i<dim;i++)to[i]*=sc;}
    poincare_expmap0(to,v_out,dim,&pb);
}
void wubu_gaad_init(WubuGaadEncoder* enc, int latent_dim) {
    memset(enc,0,sizeof(WubuGaadEncoder));
    enc->latent_dim=latent_dim; enc->point_dim=32;
    enc->num_spiral=generate_phi_spiral(enc->spiral_points,256,IMG_DIM,IMG_DIM);
    enc->num_regions=golden_subdivide(enc->regions,IMG_DIM,IMG_DIM,64);
    wubu_level_init(&enc->levels[0],32,0,4,1.0f,32);
    wubu_level_init(&enc->levels[1],32,1,3,1.0f,32);
    wubu_level_init(&enc->levels[2],32,2,2,1.0f,32);
    wubu_level_init(&enc->levels[3],32,3,2,1.0f,32);
    int DIM=enc->point_dim; unsigned int r=123;
    for(int i=0;i<3*DIM;i++){r=r*1103515245u+12345u;enc->proj_w[i]=((float)(r&0x7FFF)/32768.0f-0.5f)*0.1f;}
    for(int i=0;i<DIM;i++){r=r*1103515245u+12345u;enc->proj_b[i]=((float)(r&0x7FFF)/32768.0f-0.5f)*0.01f;}
    for(int i=0;i<latent_dim*DIM;i++){r=r*1103515245u+12345u;enc->out_w[i]=((float)(r&0x7FFF)/32768.0f-0.5f)*0.01f;}
    for(int i=0;i<latent_dim;i++)enc->out_b[i]=0.0f;
}
void wubu_gaad_free(WubuGaadEncoder* enc) {
    for(int l=0;l<MAX_LEVELS;l++){
        free(enc->levels[l].boundary_points);free(enc->levels[l].descriptor);
        free(enc->levels[l].flow_w1);free(enc->levels[l].flow_b1);
        free(enc->levels[l].flow_w2);free(enc->levels[l].flow_b2);
        free(enc->levels[l].rot_param);
        free(enc->levels[l].trans_w1);free(enc->levels[l].trans_b1);
        free(enc->levels[l].trans_w2);free(enc->levels[l].trans_b2);
    }
}
void wubu_gaad_encode(WubuGaadEncoder* enc, const float* image, float* latent) {
    int DIM=enc->point_dim, LDIM=enc->latent_dim;
    float rgb[3]={0.5f,0.3f,0.7f};
    float v[64],vo[64],ws[256];
    for(int d=0;d<DIM;d++){v[d]=enc->proj_b[d];for(int c=0;c<3;c++)v[d]+=enc->proj_w[d*3+c]*rgb[c];}
    /* Project into Poincare ball for level 0 */
    { PoincareBall pb0 = poincare_ball(enc->levels[0].c);
      poincare_project(v, DIM, &pb0); }
    for(int l=0;l<4;l++){wubu_level_forward(&enc->levels[l],v,vo,ws);memcpy(v,vo,DIM*sizeof(float));}
    for(int o=0;o<LDIM;o++){float s=enc->out_b[o];for(int d=0;d<DIM;d++)s+=enc->out_w[o*DIM+d]*v[d];latent[o]=s;}
}
void wubu_gaad_encode_full(WubuGaadEncoder* enc, const float* image, float* latents, int* num_latents) {
    int DIM=enc->point_dim, LDIM=enc->latent_dim;
    *num_latents=0;
    for(int p=0;p<enc->num_spiral&&p<256;p++){
        int ix=(int)enc->spiral_points[p].x, iy=(int)enc->spiral_points[p].y;
        if(ix<0)ix=0;if(ix>=IMG_DIM)ix=IMG_DIM-1;
        if(iy<0)iy=0;if(iy>=IMG_DIM)iy=IMG_DIM-1;
        float rgb[3]={image[(iy*IMG_DIM+ix)*3],image[(iy*IMG_DIM+ix)*3+1],image[(iy*IMG_DIM+ix)*3+2]};
        float v[64],vo[64],ws[256];
        for(int d=0;d<DIM;d++){v[d]=enc->proj_b[d];for(int c=0;c<3;c++)v[d]+=enc->proj_w[d*3+c]*rgb[c];}
        { PoincareBall pb0 = poincare_ball(enc->levels[0].c);
          poincare_project(v, DIM, &pb0); }
        for(int l=0;l<4;l++){wubu_level_forward(&enc->levels[l],v,vo,ws);memcpy(v,vo,DIM*sizeof(float));}
        for(int o=0;o<LDIM;o++){float s=enc->out_b[o];for(int d=0;d<DIM;d++)s+=enc->out_w[o*DIM+d]*v[d];latents[p*LDIM+o]=s;}
        (*num_latents)++;
    }
}
#ifdef GAAD_DEMO
int wubu_gaad_main(void){
    printf("GAAD WuBu Encoder - Golden Ratio Fractal Nesting\n");
    printf("=================================================\n\n");
    WubuGaadEncoder encoder;
    wubu_gaad_init(&encoder,32);
    float image[IMG_DIM*IMG_DIM*3];
    for(int y=0;y<IMG_DIM;y++)for(int x=0;x<IMG_DIM;x++){
        int i=(y*IMG_DIM+x)*3;
        image[i]=(float)x/(IMG_DIM-1);image[i+1]=(float)y/(IMG_DIM-1);image[i+2]=0.5f;
    }
    float latent[32];
    wubu_gaad_encode(&encoder,image,latent);
    printf("Single patch latent (32-dim):\n  [");
    for(int i=0;i<32;i++){printf("%.6f%s",latent[i],i<31?", ":"");if((i+1)%8==0&&i<31)printf("\n   ");}
    printf("]\n\n");
    float latents[256*32]; int nl;
    wubu_gaad_encode_full(&encoder,image,latents,&nl);
    printf("Full encode: %d patches\n",nl);
    printf("Config: spiral=%d, regions=%d, levels=4\n",encoder.num_spiral,encoder.num_regions);
    printf("Curvatures: L0=%.4f L1=%.4f L2=%.4f L3=%.4f\n",
           encoder.levels[0].c,encoder.levels[1].c,encoder.levels[2].c,encoder.levels[3].c);
    wubu_gaad_free(&encoder);
    return 0;
}

#endif /* GAAD_DEMO */
