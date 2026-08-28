/* wubuv_player.c -- SDL2 video player for WUBV container
 *
 * Plays WUBV files with:
 * - Video: YUV→RGB conversion, texture-based rendering
 * - Audio: Kodak-decoded samples via SDL audio callback
 * - A/V sync: frame-rate based timing
 *
 * Usage: ./wubuv_player file.wubv
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <SDL2/SDL.h>
#include "wubuv_audio.h"

#define WINDOW_TITLE "WuBu Player"

typedef struct {
    SDL_Window* window;
    SDL_Renderer* renderer;
    SDL_Texture* texture;
    SDL_AudioDeviceID audio_dev;
    
    WubuvAudioReader* reader;
    int width, height, fps;
    int has_audio;
    uint32_t audio_rate;
    
    /* Audio buffer */
    float audio_buffer[65536];
    int audio_len;
    int audio_pos;
    
    /* Timing */
    uint32_t frame_ms;
    uint32_t start_ticks;
    int frame_count;
} Player;

/* Audio callback: feed samples to SDL */
static void audio_callback(void* userdata, Uint8* stream, int len) {
    Player* p = (Player*)userdata;
    float* out = (float*)stream;
    int samples_needed = len / sizeof(float);
    int copied = 0;
    
    while(copied < samples_needed) {
        if(p->audio_pos >= p->audio_len) {
            /* Need more audio - decode next frame */
            int ret = wubuv_audio_read_audio(p->reader, p->audio_buffer, 65536);
            if(ret <= 0) {
                /* No more audio, fill with silence */
                memset(out + copied, 0, (samples_needed - copied) * sizeof(float));
                return;
            }
            p->audio_len = ret;
            p->audio_pos = 0;
        }
        
        int avail = p->audio_len - p->audio_pos;
        int to_copy = samples_needed - copied;
        if(to_copy > avail) to_copy = avail;
        
        memcpy(out + copied, p->audio_buffer + p->audio_pos, to_copy * sizeof(float));
        p->audio_pos += to_copy;
        copied += to_copy;
    }
}

int main(int argc, char** argv) {
    if(argc < 2) {
        printf("Usage: %s file.wubv\n", argv[0]);
        return 1;
    }
    
    Player player = {0};
    
    /* Initialize SDL */
    if(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) != 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }
    
    /* Open WUBV file */
    player.reader = wubuv_audio_reader_open(argv[1]);
    if(!player.reader) {
        fprintf(stderr, "Cannot open %s\n", argv[1]);
        SDL_Quit();
        return 1;
    }
    
    player.width = player.reader->base.hdr.width;
    player.height = player.reader->base.hdr.height;
    player.fps = player.reader->base.hdr.fps;
    player.has_audio = player.reader->has_audio;
    player.audio_rate = player.reader->base.hdr.audio_rate;
    player.frame_ms = 1000 / player.fps;
    
    printf("Playing: %dx%d @ %d fps", player.width, player.height, player.fps);
    if(player.has_audio) {
        printf(", audio @ %d Hz", player.audio_rate);
    }
    printf("\n");
    
    /* Create window */
    player.window = SDL_CreateWindow(WINDOW_TITLE,
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        player.width, player.height, SDL_WINDOW_SHOWN);
    
    player.renderer = SDL_CreateRenderer(player.window, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    
    /* Create texture for video */
    player.texture = SDL_CreateTexture(player.renderer,
        SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING,
        player.width, player.height);
    
    /* Setup audio */
    if(player.has_audio) {
        SDL_AudioSpec want, have;
        SDL_memset(&want, 0, sizeof(want));
        want.freq = player.audio_rate;
        want.format = AUDIO_F32SYS;
        want.channels = 1;
        want.samples = 2048;
        want.callback = audio_callback;
        want.userdata = &player;
        
        player.audio_dev = SDL_OpenAudioDevice(NULL, 0, &want, &have, 0);
        if(player.audio_dev == 0) {
            fprintf(stderr, "Audio open failed: %s\n", SDL_GetError());
        } else {
            SDL_PauseAudioDevice(player.audio_dev, 0); /* Start audio */
        }
    }
    
    /* Allocate frame buffer */
    uint8_t* frame = malloc(player.width * player.height * 3);
    
    /* Main loop */
    SDL_Event event;
    int running = 1;
    player.start_ticks = SDL_GetTicks();
    
    while(running) {
        /* Handle events */
        while(SDL_PollEvent(&event)) {
            if(event.type == SDL_QUIT) running = 0;
            if(event.type == SDL_KEYDOWN) {
                if(event.key.keysym.sym == SDLK_ESCAPE) running = 0;
                if(event.key.keysym.sym == SDLK_SPACE) {
                    /* Pause toggle */
                }
            }
        }
        
        /* Read and display video frame */
        int is_inter;
        int ret = wubuv_audio_read_video(player.reader, frame, &is_inter);
        if(ret != 0) {
            /* End of video */
            break;
        }
        
        /* Update texture */
        SDL_UpdateTexture(player.texture, NULL, frame, player.width * 3);
        
        /* Render */
        SDL_RenderClear(player.renderer);
        SDL_RenderCopy(player.renderer, player.texture, NULL, NULL);
        SDL_RenderPresent(player.renderer);
        
        /* Frame timing */
        player.frame_count++;
        uint32_t target_ticks = player.start_ticks + player.frame_count * player.frame_ms;
        uint32_t now = SDL_GetTicks();
        if(now < target_ticks) {
            SDL_Delay(target_ticks - now);
        }
    }
    
    /* Cleanup */
    free(frame);
    SDL_DestroyTexture(player.texture);
    SDL_DestroyRenderer(player.renderer);
    SDL_DestroyWindow(player.window);
    if(player.audio_dev > 0) SDL_CloseAudioDevice(player.audio_dev);
    wubuv_audio_reader_close(player.reader);
    SDL_Quit();
    
    printf("Playback complete.\n");
    return 0;
}
