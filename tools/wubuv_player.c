/* wubuv_player.c — SDL2 Video Player for WUBV Container
 *
 * Usage: ./wubuv_player file.wubv
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <SDL2/SDL.h>
#include "wubuv.h"

#define WINDOW_TITLE "WuBu Player"

typedef struct {
    SDL_Window* window;
    SDL_Renderer* renderer;
    SDL_Texture* texture;
    SDL_AudioDeviceID audio_dev;
    
    WubuvReader* reader;
    int width, height, fps;
    int has_audio;
    uint32_t audio_rate;
    
    /* Audio buffer */
    float audio_buffer[262144];
    int audio_len;
    int audio_pos;
    
    /* Timing */
    uint32_t frame_ms;
    uint32_t start_ticks;
    int frame_count;
    int paused;
} Player;

/* Audio callback */
static void audio_callback(void* userdata, Uint8* stream, int len) {
    Player* p = (Player*)userdata;
    float* out = (float*)stream;
    int samples_needed = len / sizeof(float);
    int copied = 0;
    
    while (copied < samples_needed) {
        if (p->audio_pos >= p->audio_len) {
            uint8_t type;
            uint8_t* data;
            uint32_t size;
            
            if (wubuv_read_frame(p->reader, &type, &data, &size) != 0 ||
                type != 2) { /* WUBUV_FRAME_AUDIO */
                memset(out + copied, 0, (samples_needed - copied) * sizeof(float));
                return;
            }
            
            uint16_t num_samples;
            memcpy(&num_samples, data, 2);
            p->audio_len = num_samples;
            p->audio_pos = 0;
            memcpy(p->audio_buffer, data + 2, num_samples * sizeof(float));
            free(data);
        }
        
        int avail = p->audio_len - p->audio_pos;
        int to_copy = samples_needed - copied;
        if (to_copy > avail) to_copy = avail;
        
        memcpy(out + copied, p->audio_buffer + p->audio_pos, to_copy * sizeof(float));
        p->audio_pos += to_copy;
        copied += to_copy;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s file.wubv\n", argv[0]);
        return 1;
    }
    
    Player player = {0};
    
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) != 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }
    
    player.reader = wubuv_reader_open(argv[1]);
    if (!player.reader) {
        fprintf(stderr, "Cannot open %s\n", argv[1]);
        SDL_Quit();
        return 1;
    }
    
    player.width = player.reader->hdr.width;
    player.height = player.reader->hdr.height;
    player.fps = player.reader->hdr.fps;
    player.has_audio = (player.reader->hdr.flags & 1) != 0;
    player.audio_rate = player.reader->hdr.audio_rate;
    player.frame_ms = 1000 / player.fps;
    
    printf("Playing: %s\n", argv[1]);
    printf("  Resolution: %dx%d @ %d fps\n", player.width, player.height, player.fps);
    if (player.has_audio) {
        printf("  Audio: %d Hz\n", player.audio_rate);
    }
    printf("  Frames: %d\n", player.reader->hdr.frame_count);
    printf("\nControls: ESC=Quit, SPACE=Pause, F=Fullscreen\n\n");
    
    player.window = SDL_CreateWindow(WINDOW_TITLE,
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        player.width, player.height, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
    
    player.renderer = SDL_CreateRenderer(player.window, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    
    player.texture = SDL_CreateTexture(player.renderer,
        SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING,
        player.width, player.height);
    
    if (player.has_audio) {
        SDL_AudioSpec want, have;
        SDL_memset(&want, 0, sizeof(want));
        want.freq = player.audio_rate;
        want.format = AUDIO_F32SYS;
        want.channels = player.reader->hdr.audio_channels;
        want.samples = 2048;
        want.callback = audio_callback;
        want.userdata = &player;
        
        player.audio_dev = SDL_OpenAudioDevice(NULL, 0, &want, &have, 0);
        if (player.audio_dev > 0) {
            SDL_PauseAudioDevice(player.audio_dev, 0);
        }
    }
    
    SDL_Event event;
    int running = 1;
    player.start_ticks = SDL_GetTicks();
    
    uint8_t* rgb = malloc(player.width * player.height * 3);
    
    while (running) {
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    running = 0;
                    break;
                case SDL_KEYDOWN:
                    switch (event.key.keysym.sym) {
                        case SDLK_ESCAPE:
                            running = 0;
                            break;
                        case SDLK_SPACE:
                            player.paused = !player.paused;
                            if (player.audio_dev > 0) {
                                SDL_PauseAudioDevice(player.audio_dev, player.paused);
                            }
                            break;
                        case SDLK_f: {
                            Uint32 flags = SDL_GetWindowFlags(player.window);
                            SDL_SetWindowFullscreen(player.window,
                                (flags & SDL_WINDOW_FULLSCREEN) ? 0 : SDL_WINDOW_FULLSCREEN);
                            break;
                        }
                    }
                    break;
            }
        }
        
        if (!player.paused) {
            uint8_t type;
            uint8_t* data;
            uint32_t size;
            
            int ret = wubuv_read_frame(player.reader, &type, &data, &size);
            if (ret != 0) break;
            
            if (type == 0 || type == 1) { /* KEY or INTER */
                uint16_t w, h;
                memcpy(&w, data, 2);
                memcpy(&h, data + 2, 2);
                
                /* Convert Y to RGB (grayscale for now) */
                for (int i = 0; i < w * h; i++) {
                    uint8_t y = data[ 4 + i];
                    rgb[i * 3 + 0] = y;
                    rgb[i * 3 + 1] = y;
                    rgb[i * 3 + 2] = y;
                }
                
                SDL_UpdateTexture(player.texture, NULL, rgb, w * 3);
                SDL_RenderClear(player.renderer);
                SDL_RenderCopy(player.renderer, player.texture, NULL, NULL);
                SDL_RenderPresent(player.renderer);
                
                player.frame_count++;
                uint32_t target = player.start_ticks + player.frame_count * player.frame_ms;
                uint32_t now = SDL_GetTicks();
                if (now < target) SDL_Delay(target - now);
            } else if (type == 2) { /* AUDIO */
                /* Handled by callback */
            }
            free(data);
        } else {
            SDL_Delay(10);
        }
    }
    
    free(rgb);
    SDL_DestroyTexture(player.texture);
    SDL_DestroyRenderer(player.renderer);
    SDL_DestroyWindow(player.window);
    if (player.audio_dev > 0) SDL_CloseAudioDevice(player.audio_dev);
    wubuv_reader_close(player.reader);
    SDL_Quit();
    
    printf("Played %d frames\n", player.frame_count);
    return 0;
}
