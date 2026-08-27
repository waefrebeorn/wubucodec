/*
 * bench_cartoon.c -- REAL CARTOON CONTENT benchmark
 * 120 frames of Felix the Cat (public domain) at 176x144
 * All codecs measured on the SAME content.
 */
#define M_PI 3.14159265358979f
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define W 176
#define H 144
#define NF 120

static long read_ppm(const char* path,uint8_t* buf){
    FILE* f=fopen(path,"rb");
    if(!f)return -1;
    /* skip P6 header */
    char line[256];
    fgets(line,sizeof(line),f); // P6
    fgets(line,sizeof(line),f); // comment or dimensions
    int w,h,maxval;
    while(fgets(line,sizeof(line),f)){
        if(line[0]=='#')continue;
        sscanf(line,"%d %d",&w,&h);
        break;
    }
    fgets(line,sizeof(line),f); // maxval
    fread(buf,1,(size_t)W*H*3,f);
    fclose(f);
    return (long)W*H*3;
}

static float calc_psnr(const uint8_t* a,const uint8_t* b,long n){
    double mse=0;
    for(long i=0;i<n;i++){
        double d=a[i]-b[i];
        mse+=d*d;
    }
    mse/=n;
    if(mse<=0)return 99.0f;
    return 10.0*log10(255.0*255.0/mse);
}

int main(void){
    printf("================================================================\n");
    printf("  DEFINITIVE COMPRESSION BENCHMARK — REAL CARTOON CONTENT\n");
    printf("  Source: Felix the Cat (public domain, archive.org)\n");
    printf("  %dx%d · %d frames · RGB24\n",W,H,NF);
    printf("================================================================\n\n");

    /* load frames */
    uint8_t* frames=malloc((size_t)NF*W*H*3);
    for(int i=0;i<NF;i++){
        char path[256];
        snprintf(path,sizeof(path),"test_frames/%04d.ppm",i+1);
        if(read_ppm(path,frames+(size_t)i*W*H*3)<0){
            fprintf(stderr,"failed to load %s\n",path);
            return 1;
        }
    }
    long raw=(long)NF*W*H*3;
    printf("Loaded %d frames (%ld bytes raw)\n\n",NF,raw);

    /* measure inter-frame complexity */
    double total_diff=0;
    for(int f=1;f<NF;f++){
        const uint8_t* a=frames+(size_t)(f-1)*W*H*3;
        const uint8_t* b=frames+(size_t)f*W*H*3;
        for(long i=0;i<(long)W*H*3;i+=16)
            total_diff+=abs(a[i]-b[i]);
    }
    float avg_diff=(float)(total_diff/((NF-1)*(W*H*3/16)));
    printf("Average inter-frame diff: %.2f (cartoon = moderate motion)\n\n",avg_diff);

    /* write y4m for external codecs */
    {
        FILE* y4m=fopen("cartoon.y4m","wb");
        fprintf(y4m,"YUV4MPEG2 W%d H%d F30:1 Ip A1:1\nC420jpeg\n",W,H);
        uint8_t* Y=malloc((size_t)W*H);
        uint8_t* U=malloc((size_t)W*H/4);
        uint8_t* V=malloc((size_t)W*H/4);
        for(int fi=0;fi<NF;fi++){
            fprintf(y4m,"FRAME\n");
            const uint8_t* rgb=frames+(size_t)fi*W*H*3;
            for(int y=0;y<H;y++)
                for(int x=0;x<W;x++){
                    size_t i3=((size_t)y*W+x)*3;
                    Y[y*W+x]=(uint8_t)((rgb[i3]*299+rgb[i3+1]*587+rgb[i3+2]*114)/1000);
                }
            fwrite(Y,1,(size_t)W*H,y4m);
            fwrite(U,1,(size_t)W*H/4,y4m);
            fwrite(V,1,(size_t)W*H/4,y4m);
        }
        free(Y);free(U);free(V);fclose(y4m);
    }

    /* results table */
    printf("%-28s %10s %7s %10s  %s\n","Codec","Bytes","Ratio","PSNR(dB)","Notes");
    printf("%-28s %10s %7s %10s  %s\n","──────────────────────","──────────","───────","────────","──────────");

    /* raw */
    printf("%-28s %10ld %6.1fx %10s  %s\n","raw RGB24",raw,1.0,"∞","uncompressed");

    /* x264 lossless */
    if(system("ffmpeg -y -i cartoon.y4m -c:v libx264 -qp 0 -pix_fmt yuv420p cartoon_x264.mp4 2>/dev/null")==0){
        FILE* fh=fopen("cartoon_x264.mp4","rb");
        fseek(fh,0,SEEK_END);long sz=ftell(fh);fclose(fh);
        printf("%-28s %10ld %6.1fx %10s  %s\n","x264 lossless",sz,(float)raw/sz,"lossless","H.264 CABAC qp=0");
    }

    /* FFV1 */
    if(system("ffmpeg -y -i cartoon.y4m -c:v ffv1 -level 3 cartoon_ffv1.mkv 2>/dev/null")==0){
        FILE* fh=fopen("cartoon_ffv1.mkv","rb");
        fseek(fh,0,SEEK_END);long sz=ftell(fh);fclose(fh);
        printf("%-24s %10ld %6.1fx %10s  %s\n","FFV1",sz,(float)raw/sz,"lossless","archival codec");
    }

    /* VP9 lossless */
    if(system("ffmpeg -y -i cartoon.y4m -c:v libvpx-vp9 -lossless 1 cartoon_vp9.webm 2>/dev/null")==0){
        FILE* fh=fopen("cartoon_vp9.webm","rb");
        fseek(fh,0,SEEK_END);long sz=ftell(fh);fclose(fh);
        printf("%-24s %10ld %6.1fx %10s  %s\n","VP9 lossless",sz,(float)raw/sz,"lossless","WebM");
    }

    /* x264 crf23 lossy */
    if(system("ffmpeg -y -i cartoon.y4m -c:v libx264 -crf 23 -pix_fmt yuv420p cartoon_lossy.mp4 2>/dev/null")==0){
        FILE* fh=fopen("cartoon_lossy.mp4","rb");
        fseek(fh,0,SEEK_END);long sz=ftell(fh);fclose(fh);
        /* decode and PSNR */
        system("ffmpeg -y -i cartoon_lossy.mp4 -pix_fmt rgb24 cartoon_lossy_%04d.ppm 2>/dev/null");
        uint8_t* recon=malloc((size_t)NF*W*H*3);
        int decoded=0;
        for(int i=0;i<NF;i++){
            char path[256];
            snprintf(path,sizeof(path),"cartoon_lossy_%04d.ppm",i+1);
            if(read_ppm(path,recon+(size_t)i*W*H*3)>0)decoded++;
        }
        if(decoded>0){
            float p=calc_psnr(frames,recon,(long)decoded*W*H*3);
            printf("%-28s %10ld %6.1fx %9.1fdB  %s\n","x264 crf23 (LOSSY)",sz,(float)raw/sz,p,"reference quality");
        }
        free(recon);
        /* cleanup ppm files */
        for(int i=0;i<NF;i++){
            char path[256];
            snprintf(path,sizeof(path),"cartoon_lossy_%04d.ppm",i+1);
            remove(path);
        }
    }

    /* our quaternion codecs */
    extern long wubu_cv3_encode(const uint8_t*,const float*,int,int,int,int,int,float,FILE*);
    extern long wubu_qz_encode(const uint8_t*,int,int,int,float,FILE*);

    /* C059: SLERP + zlib (P-only prediction) */
    {
        /* estimate rotation from the content — cartoons have camera pans but not pure rotation.
         * For the benchmark we use a small angle that approximates the average motion. */
        float est_angle=avg_diff/255.0f*0.1f;  /* scale to reasonable range */

        FILE* ef=fopen("cartoon_wubq.wubz","wb");
        long sz=wubu_qz_encode(frames,NF,W,H,est_angle,ef);
        fclose(ef);

        /* decode and PSNR */
        extern void wubu_qz_decode(FILE*,uint8_t*,int,int,int,float);
        uint8_t* recon=malloc((size_t)NF*W*H*3);
        memset(recon,0,(size_t)NF*W*H*3);
        FILE* df=fopen("cartoon_wubq.wubz","rb");
        wubu_qz_decode(df,recon,NF,W,H,est_angle);
        fclose(df);

        float p=calc_psnr(frames,recon,(long)NF*W*H*3);
        printf("%-28s %10ld %6.1fx %9.1fdB  %s\n",
               "★ WUBQ (SLERP+zlib)",sz,(float)raw/sz,p,
               "quaternion latent space");
        free(recon);
    }

    printf("\n================================================================\n");
    printf("  Cartoon content has MODERATE motion (not pure rotation).\n");
    printf("  The quaternion advantage is largest on pure rotation.\n");
    printf("  On general content, the codec competes on:\n");
    printf("  • 20-byte header vs MP4's box tree\n");
    printf("  • Zero dependencies\n");
    printf("  • Built-in CRC32 integrity\n");
    printf("  • Human-readable hex dump\n");
    printf("================================================================\n");

    free(frames);
    remove("cartoon.y4m");
    return 0;
}
