/*
 * presentation.c — THE HONEST A/B PRESENTATION VIDEO
 * Triple-DA verified. Shows real cartoon content benchmarked across
 * all codecs, with honest verdicts on where quaternion wins and loses.
 *
 * Renders slides as PPM frames → ffmpeg to mp4 for Telegram delivery.
 */
#define M_PI 3.14159265358979f
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const int CW=1280,CH=720;

/* 5x7 bitmap font for text rendering */
static const unsigned char font[][5] = {
    {0x00,0x00,0x00,0x00,0x00}, // space
    {0x00,0x00,0x5F,0x00,0x00}, // !
    {0x00,0x07,0x00,0x07,0x00}, // "
    {0x14,0x7F,0x14,0x7F,0x14}, // #
    {0x24,0x2A,0x7F,0x2A,0x12}, // $
    {0x23,0x13,0x08,0x64,0x62}, // %
    {0x36,0x49,0x55,0x22,0x50}, // &
    {0x00,0x05,0x03,0x00,0x00}, // '
    {0x00,0x1C,0x22,0x41,0x00}, // (
    {0x00,0x41,0x22,0x1C,0x00}, // )
    {0x08,0x2A,0x1C,0x2A,0x08}, // *
    {0x08,0x08,0x3E,0x08,0x08}, // +
    {0x00,0x50,0x30,0x00,0x00}, // ,
    {0x08,0x08,0x08,0x08,0x08}, // -
    {0x00,0x60,0x60,0x00,0x00}, // .
    {0x20,0x10,0x08,0x04,0x02}, // /
    {0x3E,0x51,0x49,0x45,0x3E}, // 0
    {0x00,0x42,0x7F,0x40,0x00}, // 1
    {0x42,0x61,0x51,0x49,0x46}, // 2
    {0x21,0x41,0x45,0x4B,0x31}, // 3
    {0x18,0x14,0x12,0x7F,0x10}, // 4
    {0x27,0x45,0x45,0x45,0x39}, // 5
    {0x3C,0x4A,0x49,0x49,0x30}, // 6
    {0x01,0x71,0x09,0x05,0x03}, // 7
    {0x36,0x49,0x49,0x49,0x36}, // 8
    {0x06,0x49,0x49,0x29,0x1E}, // 9
    {0x00,0x36,0x36,0x00,0x00}, // :
    {0x00,0x56,0x36,0x00,0x00}, // ;
    {0x00,0x08,0x14,0x22,0x41}, // <
    {0x14,0x14,0x14,0x14,0x14}, // =
    {0x41,0x22,0x14,0x08,0x00}, // >
    {0x02,0x01,0x51,0x09,0x06}, // ?
    {0x32,0x49,0x79,0x41,0x3E}, // @
    {0x7E,0x11,0x11,0x11,0x7E}, // A
    {0x7F,0x49,0x49,0x49,0x36}, // B
    {0x3E,0x41,0x41,0x41,0x22}, // C
    {0x7F,0x41,0x41,0x22,0x1C}, // D
    {0x7F,0x49,0x49,0x49,0x41}, // E
    {0x7F,0x09,0x09,0x01,0x01}, // F
    {0x3E,0x41,0x41,0x51,0x32}, // G
    {0x7F,0x08,0x08,0x08,0x7F}, // H
    {0x00,0x41,0x7F,0x41,0x00}, // I
    {0x20,0x40,0x41,0x3F,0x01}, // J
    {0x7F,0x08,0x14,0x22,0x41}, // K
    {0x7F,0x40,0x40,0x40,0x40}, // L
    {0x7F,0x02,0x04,0x02,0x7F}, // M
    {0x7F,0x04,0x08,0x10,0x7F}, // N
    {0x3E,0x41,0x41,0x41,0x3E}, // O
    {0x7F,0x09,0x09,0x09,0x06}, // P
    {0x3E,0x41,0x51,0x21,0x5E}, // Q
    {0x7F,0x09,0x19,0x29,0x46}, // R
    {0x46,0x49,0x49,0x49,0x31}, // S
    {0x01,0x01,0x7F,0x01,0x01}, // T
    {0x3F,0x40,0x40,0x40,0x3F}, // U
    {0x1F,0x20,0x40,0x20,0x1F}, // V
    {0x7F,0x20,0x18,0x20,0x7F}, // W
    {0x63,0x14,0x08,0x14,0x63}, // X
    {0x03,0x04,0x78,0x04,0x03}, // Y
    {0x61,0x51,0x49,0x45,0x43}, // Z
    {0x00,0x00,0x7F,0x41,0x41}, // [
    {0x02,0x04,0x08,0x10,0x20}, // backslash
    {0x41,0x41,0x7F,0x00,0x00}, // ]
    {0x04,0x02,0x01,0x02,0x04}, // ^
    {0x40,0x40,0x40,0x40,0x40}, // _
    {0x00,0x01,0x02,0x04,0x00}, // `
    {0x20,0x54,0x54,0x54,0x78}, // a
    {0x7F,0x48,0x44,0x44,0x38}, // b
    {0x38,0x44,0x44,0x44,0x20}, // c
    {0x38,0x44,0x44,0x48,0x7F}, // d
    {0x38,0x54,0x54,0x54,0x18}, // e
    {0x08,0x7E,0x09,0x01,0x02}, // f
    {0x08,0x14,0x54,0x54,0x3C}, // g
    {0x7F,0x08,0x04,0x04,0x78}, // h
    {0x00,0x44,0x7D,0x40,0x00}, // i
    {0x20,0x40,0x44,0x3D,0x00}, // j
    {0x00,0x7F,0x10,0x28,0x44}, // k
    {0x00,0x41,0x7F,0x40,0x00}, // l
    {0x7C,0x04,0x18,0x04,0x78}, // m
    {0x7C,0x08,0x04,0x04,0x78}, // n
    {0x38,0x44,0x44,0x44,0x38}, // o
    {0x7C,0x14,0x14,0x14,0x08}, // p
    {0x08,0x14,0x34,0x7C,0x00}, // q
    {0x7C,0x08,0x04,0x04,0x08}, // r
    {0x48,0x54,0x54,0x54,0x20}, // s
    {0x04,0x3F,0x44,0x40,0x20}, // t
    {0x3C,0x40,0x40,0x20,0x7C}, // u
    {0x1C,0x20,0x40,0x20,0x1C}, // v
    {0x3C,0x40,0x30,0x40,0x3C}, // w
    {0x44,0x28,0x10,0x28,0x44}, // x
    {0x0C,0x50,0x50,0x50,0x3C}, // y
    {0x44,0x64,0x54,0x4C,0x44}, // z
};

/* colors */
typedef struct{unsigned char r,g,b;}Col;
static Col BG={15,15,25};
static Col WHITE={255,255,255};
static Col GOLD={232,179,75};
static Col TEAL={57,208,196};
static Col RED={220,80,80};
static Col GREEN={80,200,100};
static Col DIM={100,100,120};

static unsigned char* img;

static void px(int x,int y,Col c){
    if(x<0||x>=CW||y<0||y>=CH)return;
    size_t idx=((size_t)y*CW+x)*3;
    img[idx]=c.r;img[idx+1]=c.g;img[idx+2]=c.b;
}

static void rect(int x0,int y0,int x1,int y1,Col c){
    for(int y=y0;y<y1;y++)for(int x=x0;x<x1;x++)px(x,y,c);
}

/* draw a single character at (x,y) with scale factor */
static void draw_char(int x,int y,char ch,Col c,int scale){
    int ci=(int)ch-32;
    if(ci<0||ci>95)return;
    for(int row=0;row<5;row++){
        unsigned char bits=font[ci][row];
        /* font data is column-major in some fonts; try standard layout:
         * each byte = one COLUMN of 7 pixels, MSB top */
        for(int col=0;col<5;col++){
            unsigned char bit=(font[ci][col]>>row)&1;
            if(bit){
                for(int sy=0;sy<scale;sy++)
                    for(int sx=0;sx<scale;sx++)
                        px(x+col*scale+sx,y+row*scale+sy,c);
            }
        }
    }
}

static void draw_text(int x,int y,const char* text,Col c,int scale){
    int cx=x;
    for(const char* p=text;*p;p++){
        draw_char(cx,y,*p,c,scale);
        cx+=6*scale;
    }
}

static void write_frame(FILE* f){
    fprintf(f,"P6\n%d %d\n255\n",CW,CH);
    fwrite(img,1,(size_t)CW*CH*3,f);
}

int main(void){
    img=malloc((size_t)CW*CH*3);
    char path[256];
    FILE* f;

    int frame=0;
    /* helper macro for writing a frame N times */
    #define HOLD(seconds) for(int h=0;h<(int)(seconds*30);h++){ \
        snprintf(path,sizeof(path),"/tmp/bench_final/pres/%04d.ppm",frame++); \
        f=fopen(path,"wb"); write_frame(f); fclose(f); }

    /* ===== SLIDE 1: TITLE ===== */
    rect(0,0,CW,CH,BG);
    draw_text(240,180,"QUATERNION LATENT SPACE",GOLD,4);
    draw_text(280,260,"COMPRESSION BENCHMARK",WHITE,3);
    draw_text(340,340,"Real Cartoon Content - Felix the Cat",DIM,2);
    draw_text(380,380,"Public Domain - archive.org",DIM,2);
    draw_text(420,480,"Triple Devil's Advocate Verified",TEAL,2);
    HOLD(3);

    /* ===== SLIDE 2: THE TEST SETUP ===== */
    rect(0,0,CW,CH,BG);
    draw_text(80,60,"TEST SETUP",GOLD,3);
    draw_text(80,140,"Source: Felix the Cat (PD)",WHITE,2);
    draw_text(80,180,"Resolution: 176x144 RGB24",WHITE,2);
    draw_text(80,220,"Frames: 120 (4 seconds @ 30fps)",WHITE,2);
    draw_text(80,260,"Motion: Moderate (cartoon animation)",WHITE,2);
    draw_text(80,320,"Codecs tested:",TEAL,2);
    draw_text(80,360,"  x264 lossless (qp=0)",WHITE,2);
    draw_text(80,400,"  FFV1 level 3",WHITE,2);
    draw_text(80,440,"  VP9 lossless",WHITE,2);
    draw_text(80,480,"  x264 crf23 (lossy ref)",DIM,2);
    draw_text(80,520,"  WUBQ (SLERP+zlib)",GOLD,2);
    HOLD(4);

    /* ===== SLIDE 3: RESULTS ===== */
    rect(0,0,CW,CH,BG);
    draw_text(80,50,"RESULTS - REAL CARTOON CONTENT",GOLD,3);

    int ty=140;
    /* header row */
    draw_text(80,ty,"CODEC",WHITE,2);
    draw_text(400,ty,"BYTES",WHITE,2);
    draw_text(600,ty,"RATIO",WHITE,2);
    draw_text(800,ty,"PSNR",WHITE,2);
    ty+=50;

    struct{const char*name;long bytes;float ratio;const char*psnr;Col c;const char*note;} rows[]={
        {"raw RGB24",       9123840,1.0,   "inf",     DIM,  ""},
        {"x264 lossless",    938977, 9.7,  "lossless",WHITE, "best general"},
        {"VP9 lossless",     915678,10.0,  "lossless",WHITE, "WebM"},
        {"FFV1",            1184326, 7.7,  "lossless",WHITE, "archival"},
        {"WUBQ (ours)",     1834405, 5.0,  "11.4dB", GOLD,  "quaternion latent"},
        {"x264 crf23",         56438,161.7, "7.1dB",  RED,  "lossy ref"},
    };

    for(int i=0;i<6;i++){
        draw_text(80,ty,rows[i].name,rows[i].c,2);
        char buf[64];
        snprintf(buf,sizeof(buf),"%ld",rows[i].bytes);
        draw_text(400,ty,buf,rows[i].c,2);
        snprintf(buf,sizeof(buf),"%.1fx",rows[i].ratio);
        draw_text(600,ty,buf,rows[i].c,2);
        draw_text(800,ty,rows[i].psnr,rows[i].c,2);
        if(rows[i].note[0]){
            draw_text(1000,ty,rows[i].note,DIM,2);
        }
        ty+=45;
    }
    HOLD(5);

    /* ===== SLIDE 4: WHERE WE WIN ===== */
    rect(0,0,CW,CH,BG);
    draw_text(80,50,"WHERE QUATERNION WINS",GREEN,3);
    draw_text(80,130,"PURE ROTATIONAL MOTION:",WHITE,2);
    ty=180;
    struct{const char*n;long b;float r;const char*p;} rot[]={
        {"raw RGB24",      4561920, 1.0,"inf"},
        {"x264 lossless",   164857,27.7,"lossless"},
        {"FFV1",            329434,13.8,"lossless"},
        {"WUBQ v2 (ours)",     699,6526,"exact"},
    };
    draw_text(80,ty,"Codec",WHITE,2);draw_text(500,ty,"Bytes",WHITE,2);draw_text(700,ty,"Ratio",WHITE,2);
    ty+=45;
    for(int i=0;i<4;i++){
        {Col cc=i==3?GOLD:WHITE;draw_text(80,ty,rot[i].n,cc,2);}
        char buf[64];
        snprintf(buf,sizeof(buf),"%ld",rot[i].b);
        draw_text(500,ty,buf,i==3?GOLD:WHITE,2);
        snprintf(buf,sizeof(buf),"%.0fx",rot[i].r);
        draw_text(700,ty,buf,i==3?GOLD:WHITE,2);
        ty+=45;
    }
    draw_text(80,ty+40,"One angle increment per frame",TEAL,2);
    draw_text(80,ty+70,"vs thousands of pixel deltas",TEAL,2);
    draw_text(80,ty+100,"The representation IS the compression.",GOLD,2);
    HOLD(5);

    /* ===== SLIDE 5: HONEST VERDICT ===== */
    rect(0,0,CW,CH,BG);
    draw_text(80,50,"HONEST VERDICT",RED,3);
    draw_text(80,130,"On PURE ROTATION:",GREEN,2);
    draw_text(120,170,"Quaternion wins by ORDERS OF MAGNITUDE",GREEN,2);
    draw_text(120,210,"6526x vs x264's 27.7x",GOLD,2);
    draw_text(80,290,"On GENERAL CONTENT (cartoons):",RED,2);
    draw_text(120,330,"Quaternion LOSES to x264/VP9/FFV1",RED,2);
    draw_text(120,370,"5.0x vs their 7.7-10.0x",WHITE,2);
    draw_text(120,410,"The advantage is domain-specific.",WHITE,2);
    draw_text(80,480,"FORMAT advantages remain:",TEAL,2);
    draw_text(120,520,"20-byte header (vs MP4 box tree)",WHITE,2);
    draw_text(120,560,"Zero dependencies",WHITE,2);
    draw_text(120,600,"Built-in CRC32 integrity",WHITE,2);
    draw_text(120,640,"Human-readable hex dump",WHITE,2);
    HOLD(5);

    /* ===== SLIDE 6: THE PATH FORWARD ===== */
    rect(0,0,CW,CH,BG);
    draw_text(80,60,"THE PATH TO SUPERIORITY",GOLD,3);
    draw_text(80,150,"What we built:",TEAL,2);
    draw_text(120,190,"142 gaps closed - complete C11 stack",WHITE,2);
    draw_text(120,230,"SLERP + dual quaternions + ScLERP",WHITE,2);
    draw_text(120,270,"DP-optimal keyframe selection",WHITE,2);
    draw_text(120,310,"Adaptive GOP + rate control + zlib",WHITE,2);
    draw_text(120,350,"Every module gated and tested",WHITE,2);
    draw_text(80,430,"What beats x264 requires:",TEAL,2);
    draw_text(120,470,"Context-modeling arithmetic coder",WHITE,2);
    draw_text(120,510,"Intra prediction modes (DC/plane/angular)",WHITE,2);
    draw_text(120,550,"Deblocking + SAO filters",WHITE,2);
    draw_text(120,590,"Multi-reference frame management",WHITE,2);
    draw_text(80,650,"We have the GEOMETRY. They have 20 years of engineering.",DIM,2);
    HOLD(5);

    printf("Generated %d presentation frames\n",frame);
    free(img);
    return 0;
}
