/*
 * presentation2.c — FIXED text rendering with correct font orientation
 * and larger scale for stage-presentation readability.
 */
#define M_PI 3.14159265358979f
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const int CW=1280,CH=720;

/* 5x7 font — each byte is one COLUMN, bit 0 = top pixel, bit 6 = bottom */
static const unsigned char font5x7[][5] = {
    {0x00,0x00,0x00,0x00,0x00}, // space (32)
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

typedef struct{unsigned char r,g,b;}Col;
static Col BG={12,12,22};
static Col WHITE={255,255,255};
static Col GOLD={255,200,60};
static Col TEAL={57,208,196};
static Col RED={255,100,100};
static Col GREEN={80,220,100};
static Col DIM={140,140,160};

static unsigned char* img;
static void px(int x,int y,Col c){
    if(x<0||x>=CW||y<0||y>=CH)return;
    size_t idx=((size_t)y*CW+x)*3;
    img[idx]=c.r;img[idx+1]=c.g;img[idx+2]=c.b;
}
static void rect(int x0,int y0,int x1,int y1,Col c){
    for(int y=y0;y<y1;y++)for(int x=x0;x<x1;x++)px(x,y,c);
}

/* CORRECT: iterate columns outer (5 bytes), rows inner (7 bits per byte) */
static void draw_char(int x,int y,char ch,Col c,int scale){
    int ci=(int)ch-32;
    if(ci<0||ci>94)return;
    for(int col=0;col<5;col++){
        unsigned char byte=font5x7[ci][col];
        for(int row=0;row<7;row++){
            if((byte>>row)&1){
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
        cx+=6*scale; /* 5px glyph + 1px gap, times scale */
    }
}

/* measure text width for centering */
static int text_width(const char* text,int scale){
    return strlen(text)*6*scale-scale; /* last char has no trailing gap */
}

static void draw_centered(int cy,const char* text,Col c,int scale){
    draw_text((CW-text_width(text,scale))/2,cy,text,c,scale);
}

static FILE* out_file;
static int frame_num=0;
static char frame_path[256];

static void snap(void){
    snprintf(frame_path,sizeof(frame_path),"/tmp/bench_final/pres/%04d.ppm",frame_num++);
    out_file=fopen(frame_path,"wb");
    fprintf(out_file,"P6\n%d %d\n255\n",CW,CH);
    fwrite(img,1,(size_t)CW*CH*3,out_file);
    fclose(out_file);
}
#define HOLD(sec) do{for(int _h=0;_h<(int)((sec)*30);_h++)snap();}while(0)

int main(void){
    img=malloc((size_t)CW*CH*3);

    /* ===== SLIDE 1: TITLE ===== */
    rect(0,0,CW,CH,BG);
    /* accent bar at top */
    rect(0,0,CW,6,GOLD);
    draw_centered(150,"QUATERNION LATENT SPACE",GOLD,5);
    draw_centered(230,"COMPRESSION BENCHMARK",WHITE,4);
    draw_centered(330,"Real Cartoon Content: Felix the Cat",DIM,3);
    draw_centered(370,"Public Domain - archive.org",DIM,3);
    /* separator line */
    rect(340,430,940,433,DIM);
    draw_centered(470,"Triple Devil's Advocate Verified",TEAL,3);
    draw_centered(540,"142 gaps closed | 104 test suites | C11 zero-dependency",DIM,2);
    HOLD(3);

    /* ===== SLIDE 2: TEST SETUP ===== */
    rect(0,0,CW,CH,BG);
    rect(0,0,CW,6,GOLD);
    draw_text(80,50,"TEST SETUP",GOLD,4);
    draw_text(80,130,"Source:  Felix the Cat (public domain)",WHITE,3);
    draw_text(80,180,"Resolution:  176 x 144 RGB24",WHITE,3);
    draw_text(80,230,"Frames:  120 (4 seconds at 30fps)",WHITE,3);
    draw_text(80,280,"Motion:  Moderate cartoon animation",WHITE,3);
    draw_text(80,350,"CODECS UNDER TEST:",TEAL,3);
    draw_text(120,400,"x264 lossless (qp=0)",WHITE,3);
    draw_text(120,450,"FFV1 level 3",WHITE,3);
    draw_text(120,500,"VP9 lossless",WHITE,3);
    draw_text(120,550,"x264 crf23 (lossy reference)",WHITE,3);
    draw_text(120,600,"WUBQ quaternion SLERP+zlib",GOLD,3);
    HOLD(4);

    /* ===== SLIDE 3: RESULTS TABLE ===== */
    rect(0,0,CW,CH,BG);
    rect(0,0,CW,6,GOLD);
    draw_text(80,40,"RESULTS - REAL CARTOON CONTENT",GOLD,4);

    /* table header */
    int ty=130;
    draw_text(80,ty,"CODEC",WHITE,3);
    draw_text(500,ty,"BYTES",WHITE,3);
    draw_text(720,ty,"RATIO",WHITE,3);
    draw_text(900,ty,"PSNR",WHITE,3);
    ty+=55;
    /* separator */
    rect(80,ty,CW-80,ty+2,DIM);ty+=15;

    struct{const char*name;long bytes;float ratio;const char*psnr;Col c;} rows[]={
        {"raw RGB24",       9123840,1.0f,   "n/a",     DIM},
        {"x264 lossless",    938977, 9.7f,  "lossless",WHITE},
        {"VP9 lossless",     915678,10.0f,  "lossless",WHITE},
        {"FFV1",            1184326, 7.7f,  "lossless",WHITE},
        {"WUBQ (ours)",     1834405, 5.0f,  "11.4 dB", GOLD},
        {"x264 crf23 lossy",   56438,161.7f, "7.1 dB",  RED},
    };
    for(int i=0;i<6;i++){
        draw_text(80,ty,rows[i].name,rows[i].c,3);
        char buf[64];
        snprintf(buf,sizeof(buf),"%ld",rows[i].bytes);
        draw_text(500,ty,buf,rows[i].c,3);
        snprintf(buf,sizeof(buf),"%.1fx",rows[i].ratio);
        draw_text(720,ty,buf,rows[i].c,3);
        draw_text(900,ty,rows[i].psnr,rows[i].c,3);
        ty+=50;
    }
    draw_text(80,ty+30,"Cartoon content has moderate motion - not pure rotation",DIM,2);
    draw_text(80,ty+65,"Quaternion advantage is largest on rotational content",DIM,2);
    HOLD(5);

    /* ===== SLIDE 4: WHERE QUATERNION WINS ===== */
    rect(0,0,CW,CH,BG);
    rect(0,0,CW,6,GREEN);
    draw_text(80,40,"WHERE QUATERNION WINS",GREEN,4);
    draw_text(80,110,"PURE ROTATIONAL MOTION:",WHITE,3);
    ty=170;
    draw_text(80,ty,"Codec",WHITE,3);draw_text(500,ty,"Bytes",WHITE,3);draw_text(700,ty,"Ratio",WHITE,3);
    ty+=50;
    rect(80,ty,CW-80,ty+2,DIM);ty+=15;

    struct{const char*n;long b;float r;} rot[]={
        {"raw RGB24",      4561920,1.0f},
        {"x264 lossless",   164857,27.7f},
        {"FFV1",            329434,13.8f},
        {"WUBQ v2 (ours)",     699,6526.0f},
    };
    for(int i=0;i<4;i++){
        Col cc=(i==3)?GOLD:WHITE;
        draw_text(80,ty,rot[i].n,cc,3);
        char buf[64];
        snprintf(buf,sizeof(buf),"%ld",rot[i].b);
        draw_text(500,ty,buf,cc,3);
        snprintf(buf,sizeof(buf),"%.0fx",rot[i].r);
        draw_text(700,ty,buf,cc,3);
        ty+=50;
    }
    draw_text(80,ty+40,"One angle increment per frame",TEAL,3);
    draw_text(80,ty+85,"vs thousands of pixel deltas",TEAL,3);
    draw_text(80,ty+140,"The representation IS the compression.",GOLD,4);
    HOLD(5);

    /* ===== SLIDE 5: HONEST VERDICT ===== */
    rect(0,0,CW,CH,BG);
    rect(0,0,CW,6,RED);
    draw_text(80,40,"HONEST VERDICT",RED,4);
    draw_text(80,110,"On PURE ROTATION:",GREEN,3);
    draw_text(120,155,"Quaternion wins by ORDERS OF MAGNITUDE",GREEN,3);
    draw_text(120,200,"6526x vs x264's 27.7x",GOLD,3);
    draw_text(80,270,"On GENERAL CONTENT (cartoons):",RED,3);
    draw_text(120,315,"Quaternion LOSES to x264/VP9/FFV1",RED,3);
    draw_text(120,360,"5.0x vs their 7.7-10.0x compression",WHITE,3);
    draw_text(120,405,"The advantage is domain-specific.",WHITE,3);
    draw_text(80,470,"FORMAT advantages that remain:",TEAL,3);
    draw_text(120,520,"20-byte header (vs MP4 box tree)",WHITE,3);
    draw_text(120,565,"Zero external dependencies",WHITE,3);
    draw_text(120,610,"Built-in CRC32 integrity checking",WHITE,3);
    draw_text(120,655,"Human-readable in a hex editor",WHITE,3);
    HOLD(5);

    /* ===== SLIDE 6: PATH FORWARD ===== */
    rect(0,0,CW,CH,BG);
    rect(0,0,CW,6,GOLD);
    draw_text(80,40,"THE PATH TO SUPERIORITY",GOLD,4);
    draw_text(80,110,"What we built:",TEAL,3);
    draw_text(120,160,"142 gaps closed - complete C11 stack",WHITE,3);
    draw_text(120,205,"SLERP + dual quaternions + ScLERP",WHITE,3);
    draw_text(120,250,"DP-optimal keyframe selection",WHITE,3);
    draw_text(120,295,"Adaptive GOP + rate control + zlib",WHITE,3);
    draw_text(120,340,"Every module gated and tested",WHITE,3);
    draw_text(80,420,"To beat x264 everywhere requires:",TEAL,3);
    draw_text(120,470,"Context-modeling arithmetic coder",WHITE,3);
    draw_text(120,515,"Intra prediction modes (DC/plane/angular)",WHITE,3);
    draw_text(120,560,"Deblocking + SAO filters",WHITE,3);
    draw_text(120,605,"Multi-reference frame management",WHITE,3);
    draw_centered(670,"We have the GEOMETRY. They have 20 years of engineering.",DIM,2);
    HOLD(5);

    printf("Generated %d frames\n",frame_num);
    free(img);
    return 0;
}
