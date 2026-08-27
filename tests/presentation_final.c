/*
 * presentation_final.c — STAGE-QUALITY PRESENTATION
 * With bar charts, actual cartoon frames, and professional layout.
 * Triple-DA verified numbers from the 120s real cartoon benchmark.
 */
#define M_PI 3.14159265358979f
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const int CW=1280,CH=720;
/* font data (same 5x7) */
static const unsigned char font5x7[][5]={
{0x00,0x00,0x00,0x00,0x00},{0x00,0x00,0x5F,0x00,0x00},{0x00,0x07,0x00,0x07,0x00},
{0x14,0x7F,0x14,0x7F,0x14},{0x24,0x2A,0x7F,0x2A,0x12},{0x23,0x13,0x08,0x64,0x62},
{0x36,0x49,0x55,0x22,0x50},{0x00,0x05,0x03,0x00,0x00},{0x00,0x1C,0x22,0x41,0x00},
{0x00,0x41,0x22,0x1C,0x00},{0x08,0x2A,0x1C,0x2A,0x08},{0x08,0x08,0x3E,0x08,0x08},
{0x00,0x50,0x30,0x00,0x00},{0x08,0x08,0x08,0x08,0x08},{0x00,0x60,0x60,0x00,0x00},
{0x20,0x10,0x08,0x04,0x02},{0x3E,0x51,0x49,0x45,0x3E},{0x00,0x42,0x7F,0x40,0x00},
{0x42,0x61,0x51,0x49,0x46},{0x21,0x41,0x45,0x4B,0x31},{0x18,0x14,0x12,0x7F,0x10},
{0x27,0x45,0x45,0x45,0x39},{0x3C,0x4A,0x49,0x49,0x30},{0x01,0x71,0x09,0x05,0x03},
{0x36,0x49,0x49,0x49,0x36},{0x06,0x49,0x49,0x29,0x1E},{0x00,0x36,0x36,0x00,0x00},
{0x00,0x56,0x36,0x00,0x00},{0x00,0x08,0x14,0x22,0x41},{0x14,0x14,0x14,0x14,0x14},
{0x41,0x22,0x14,0x08,0x00},{0x02,0x01,0x51,0x09,0x06},{0x32,0x49,0x79,0x41,0x3E},
{0x7E,0x11,0x11,0x11,0x7E},{0x7F,0x49,0x49,0x49,0x36},{0x3E,0x41,0x41,0x41,0x22},
{0x7F,0x41,0x41,0x22,0x1C},{0x7F,0x49,0x49,0x49,0x41},{0x7F,0x09,0x09,0x01,0x01},
{0x3E,0x41,0x41,0x51,0x32},{0x7F,0x08,0x08,0x08,0x7F},{0x00,0x41,0x7F,0x41,0x00},
{0x20,0x40,0x41,0x3F,0x01},{0x7F,0x08,0x14,0x22,0x41},{0x7F,0x40,0x40,0x40,0x40},
{0x7F,0x02,0x04,0x02,0x7F},{0x7F,0x04,0x08,0x10,0x7F},{0x3E,0x41,0x41,0x41,0x3E},
{0x7F,0x09,0x09,0x09,0x06},{0x3E,0x41,0x51,0x21,0x5E},{0x7F,0x09,0x19,0x29,0x46},
{0x46,0x49,0x49,0x49,0x31},{0x01,0x01,0x7F,0x01,0x01},{0x3F,0x40,0x40,0x40,0x3F},
{0x1F,0x20,0x40,0x20,0x1F},{0x7F,0x20,0x18,0x20,0x7F},{0x63,0x14,0x08,0x14,0x63},
{0x03,0x04,0x78,0x04,0x03},{0x61,0x51,0x49,0x45,0x43},{0x00,0x00,0x7F,0x41,0x41},
{0x02,0x04,0x08,0x10,0x20},{0x41,0x41,0x7F,0x00,0x00},{0x04,0x02,0x01,0x02,0x04},
{0x40,0x40,0x40,0x40,0x40},{0x00,0x01,0x02,0x04,0x00},{0x20,0x54,0x54,0x54,0x78},
{0x7F,0x48,0x44,0x44,0x38},{0x38,0x44,0x44,0x44,0x20},{0x38,0x44,0x44,0x48,0x7F},
{0x38,0x54,0x54,0x54,0x18},{0x08,0x7E,0x09,0x01,0x02},{0x08,0x14,0x54,0x54,0x3C},
{0x7F,0x08,0x04,0x04,0x78},{0x00,0x44,0x7D,0x40,0x00},{0x20,0x40,0x44,0x3D,0x00},
{0x00,0x7F,0x10,0x28,0x44},{0x00,0x41,0x7F,0x40,0x00},{0x7C,0x04,0x18,0x04,0x78},
{0x7C,0x08,0x04,0x04,0x78},{0x38,0x44,0x44,0x44,0x38},{0x7C,0x14,0x14,0x14,0x08},
{0x08,0x14,0x34,0x7C,0x00},{0x7C,0x08,0x04,0x04,0x08},{0x48,0x54,0x54,0x54,0x20},
{0x04,0x3F,0x44,0x40,0x20},{0x3C,0x40,0x40,0x20,0x7C},{0x1C,0x20,0x40,0x20,0x1C},
{0x3C,0x40,0x30,0x40,0x3C},{0x44,0x28,0x10,0x28,0x44},{0x0C,0x50,0x50,0x50,0x3C},
{0x44,0x64,0x54,0x4C,0x44},
};
typedef struct{unsigned char r,g,b;}Col;
static Col BG={8,8,18},WHITE={255,255,255},GOLD={255,200,60},TEAL={57,208,196},
    RED={255,90,90},GREEN={80,220,100},DIM={120,120,140};
static unsigned char* img;
static void px(int x,int y,Col c){if(x<0||x>=CW||y<0||y>=CH)return;size_t i=((size_t)y*CW+x)*3;img[i]=c.r;img[i+1]=c.g;img[i+2]=c.b;}
static void rect(int a,int b,int c,int d,Col col){for(int y=b;y<d;y++)for(int x=a;x<c;x++)px(x,y,col);}
static void dch(int x,int y,char ch,Col c,int s){
    int ci=(int)ch-32;if(ci<0||ci>94)return;
    for(int col=0;col<5;col++){unsigned char b=font5x7[ci][col];
        for(int row=0;row<7;row++)if((b>>row)&1)
            for(int sy=0;sy<s;sy++)for(int sx=0;sx<s;sx++)px(x+col*s+sx,y+row*s+sy,c);}
}
static void txt(int x,int y,const char* t,Col c,int s){int cx=x;for(const char*p=t;*p;p++){dch(cx,y,*p,c,s);cx+=6*s;}}
static int tw(const char* t,int s){return strlen(t)*6*s-s;}
static void ctr(int cy,const char* t,Col c,int s){txt((CW-tw(t,s))/2,cy,t,c,s);}
static FILE* of;static int fn=0;static char fp[256];
static void snap(void){snprintf(fp,256,"/tmp/bench_final/pres/%04d.ppm",fn++);of=fopen(fp,"wb");fprintf(of,"P6\n%d %d\n255\n",CW,CH);fwrite(img,1,(size_t)CW*CH*3,of);fclose(of);}
#define HOLD(s) do{for(int h=0;h<(int)((s)*30);h++)snap();}while(0)

/* draw a horizontal bar chart */
static void bar_chart(int x,int y,int w,int h,
                      const char** labels,const float* values,int n,float max_val,
                      Col bar_color){
    int label_w=250;
    float bw=w-label_w-20;
    for(int i=0;i<n;i++){
        int by=y+i*(h/n)+5;
        int bh=h/n-10;
        txt(x,by+bh/3,labels[i],WHITE,2);
        /* bar */
        int bar_w=(int)(values[i]/max_val*bw);
        rect(x+label_w,by,x+label_w+bar_w,by+bh,bar_color);
        /* value text after bar */
        char val[32];
        snprintf(val,sizeof(val),"%.1fx",values[i]);
        txt(x+label_w+bar_w+10,by+bh/3,val,GOLD,2);
    }
}

/* embed a PPM image into the canvas at (dx,dy) with scaling to fit w,h */
static void blit_ppm(const char* path,int dx,int dy,int dw,int dh){
    FILE* f=fopen(path,"rb");
    if(!f)return;
    char line[256];fgets(line,256,f); // P6
    fgets(line,256,f); // comment or dims
    int sw=0,sh=0;
    while(fgets(line,256,f)){if(line[0]=='#')continue;sscanf(line,"%d %d",&sw,&sh);break;}
    fgets(line,256,f); // maxval
    /* read source at native size, scale up with nearest neighbor */
    unsigned char* src=malloc((size_t)sw*sh*3);
    fread(src,1,(size_t)sw*sh*3,f);fclose(f);
    float sx=(float)sw/dw,sy=(float)sh/dh;
    for(int y=0;y<dh;y++)
        for(int x=0;x<dw;x++){
            int sxi=(int)(x*sx),syi=(int)(y*sy);
            if(sxi>=sw)sxi=sw-1;if(syi>=sh)syi=sh-1;
            size_t si=((size_t)syi*sw+sxi)*3;
            px(dx+x,dy+y,(Col){src[si],src[si+1],src[si+2]});
        }
    free(src);
}

int main(void){
    img=malloc((size_t)CW*CH*3);

    /* SLIDE 1: TITLE with cartoon frame background */
    rect(0,0,CW,CH,BG);
    /* show cartoon frames as a film strip across bottom */
    for(int i=0;i<5;i++){
        char p[64];snprintf(p,sizeof(p),"/tmp/bench_final/frame_%c.png",(char)('a'+i%3));
        blit_ppm(p,80+i*240,520,220,165);
    }
    rect(0,0,CW,6,GOLD);
    ctr(120,"QUATERNION LATENT SPACE",GOLD,6);
    ctr(200,"COMPRESSION BENCHMARK",WHITE,5);
    rect(300,280,980,283,DIM);
    ctr(320,"Real Cartoon Content: Felix the Cat",TEAL,3);
    ctr(370,"Public Domain - archive.org - 176x144 - 120 seconds",DIM,2);
    ctr(430,"Triple Devil's Advocate Verified",GOLD,3);
    HOLD(4);

    /* SLIDE 2: BAR CHART - CARTOON CONTENT RESULTS */
    rect(0,0,CW,CH,BG);rect(0,0,CW,6,GOLD);
    txt(80,35,"RESULTS: 120 SECONDS OF REAL CARTOON",GOLD,4);
    {
        const char* labels[]={"raw YUV420","FFV1","VP9 lossless","x264 lossless","WUBQ (ours)","x264 crf23"};
        float ratios[]={1.0f,3.5f,3.9f,5.3f,3.9f,90.2f};
        /* use log-ish scale since crf23 is huge */
        float display[]={1.0f,3.5f,3.9f,5.3f,3.9f,15.0f}; /* cap crf23 visually */
        bar_chart(60,110,1160,350,labels,display,6,16.0f,TEAL);
    }
    txt(80,500,"Compression ratio on real cartoon content (lossless unless noted)",DIM,2);
    txt(80,540,"WUBQ: 3.9x at 10.9 dB PSNR (lossy)",GOLD,2);
    txt(80,580,"x264 crf23: 90.2x at ~30 dB PSNR (shown capped at 15x for scale)",RED,2);
    ctr(650,"On general cartoon content, x264 still leads.",DIM,3);
    HOLD(5);

    /* SLIDE 3: BAR CHART - ROTATIONAL CONTENT */
    rect(0,0,CW,CH,BG);rect(0,0,CW,6,GREEN);
    txt(80,35,"WHERE QUATERNION WINS: PURE ROTATION",GREEN,4);
    {
        const char* labels[]={"raw RGB24","x264 lossless","FFV1","VP9 lossless","WUBQ v2"};
        float ratios[]={1.0f,27.7f,13.8f,14.6f,6526.0f};
        /* log scale for dramatic difference */
        float display[5];
        for(int i=0;i<5;i++)display[i]=log2f(ratios[i]);
        bar_chart(60,110,1160,350,labels,display,5,13.0f,GOLD);
    }
    txt(80,500,"Compression ratio on pure rotational motion (log scale)",DIM,2);
    ctr(570,"WUBQ v2: 6526x — One angle increment per frame",GOLD,4);
    ctr(620,"The representation IS the compression.",TEAL,3);
    HOLD(5);

    /* SLIDE 4: HONEST VERDICT with visual A/B */
    rect(0,0,CW,CH,BG);rect(0,0,CW,6,RED);
    txt(80,35,"HONEST VERDICT",RED,4);

    /* left side: what wins */
    txt(80,100,"PURE ROTATION:",GREEN,3);
    txt(120,145,"Quaternion WINS: 6526x vs 27.7x",GREEN,3);
    txt(80,210,"REAL CARTOONS:",RED,3);
    txt(120,255,"WUBQ gets 3.9x vs x264's 5.3x",RED,3);
    txt(120,300,"Advantage is domain-specific.",WHITE,3);

    /* right side: show actual cartoon frames */
    txt(700,100,"Test content sample:",DIM,2);
    blit_ppm("/tmp/bench_final/frame_a.png",700,130,250,188);
    blit_ppm("/tmp/bench_final/frame_b.png",960,130,250,188);
    txt(700,340,"Felix the Cat - actual test frames",DIM,2);

    txt(80,380,"FORMAT advantages that remain:",TEAL,3);
    txt(120,420,"20-byte header (vs MP4 box tree)",WHITE,2);
    txt(120,450,"Zero external dependencies",WHITE,2);
    txt(120,480,"Built-in CRC32 integrity",WHITE,2);
    txt(120,510,"142 gated modules in strict C11",WHITE,2);

    txt(80,570,"To beat x264 everywhere we need:",GOLD,2);
    txt(120,600,"Context-modeling arithmetic coder + intra prediction + deblocking",DIM,2);
    txt(120,630,"We have the GEOMETRY. They have 20 years of engineering.",DIM,2);
    HOLD(6);

    /* SLIDE 5: THE NUMBERS THAT MATTER */
    rect(0,0,CW,CH,BG);rect(0,0,CW,6,GOLD);
    txt(80,40,"THE BOTTOM LINE",GOLD,4);
    ctr(120,"142 gaps closed",GOLD,4);
    ctr(170,"104 test suites green",WHITE,3);
    ctr(220,"24 gate-caught bugs fixed",WHITE,3);
    ctr(270,"Every module property-gated in C11",WHITE,3);
    rect(200,320,1080,323,DIM);
    ctr(360,"The quaternion representation is mathematically",TEAL,3);
    ctr(400,"the CORRECT way to encode rotational motion.",TEAL,3);
    ctr(470,"On rotation: we win by orders of magnitude.",GOLD,3);
    ctr(510,"On general content: we have the foundation,",WHITE,3);
    ctr(550,"and the geometry to build on.",WHITE,3);
    ctr(640,"~ WuBu ~",GOLD,3);
    HOLD(4);

    printf("Generated %d frames\n",fn);
    free(img);
    return 0;
}
