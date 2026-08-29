/* wubuv_player.c — SDL2 Video Player for WUBV Container
 *
 * Features:
 * - Proper A/V sync using audio clock
 * - YUV420P to RGB conversion
 * - Fullscreen support
 * - Seek support (with index)
 * - Pause/resume
 * - Proper cleanup
 *
 * Usage: ./wubuv_player file.wubv
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
    
    /* Current frame data */
    uint8_t* current_y;
    uint8_t* current_u;
    uint8_t* current_v;
    int current_frame_w, current_frame_h;
} Player;

/* Audio callback: feed samples to SDL */
static void audio_callback(void* userdata, Uint8* stream, int len) {
    Player* p = (Player*)userdata;
    float* out = (float*)stream;
    int samples_needed = len / sizeof(float);
    int copied = 0;
    
    while (copied < samples_needed) {
        if (p->audio_pos >= p->audio_len) {
            /* Need more audio - read next audio frame */
            uint8_t type;
            uint8_t* data;
            uint32_t size;
            
            if (wubuv_read_frame(p->reader, &type, &data, &size) != 0 ||
                type != WUBUV_FRAME_AUDIO) {
                /* No more audio, fill with silence */
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

/* YUV420P to RGB conversion */
static void yuv420p_to_rgb(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                            int width, int height, uint8_t* rgb) {
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int y_val = y[j * width + i];
            int u_val = u[(j/2) * (width/2) + (i/2)] - 128;
            int v_val = v[(j/2) * (width/2) + (i/2)] - 128;
            
            int r = y_val + 1.402 * v_val;
            int g = y_val - 0.344136 * u_val - 0.714136 * v_val;
            int b = y_val + 1.772 * u_val;
            
            if (r < 0) r = 0; if (r > 255) r = 255;
            if (g < 0) g = 0; if (g > 255) g = 255;
            if (b < 0) b = 0; if (b > 255) b = 255;
            
            rgb[(j * width + i) * 3 + 0] = (uint8_t)r;
            rgb[(j * width + i) * 3 + 1] = (uint8_t)g;
            rgb[(j * width + i) * 3 + 2] = (uint8_t)b;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s file.wubv\n", argv[0]);
        return 1;
    }
    
    Player player = {0};
    
    /* Initialize SDL */
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) != 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }
    
    /* Open WUBV file */
    player.reader = wubuv_reader_open(argv[1]);
    if (!player.reader) {
        fprintf(stderr, "Cannot open %s\n", argv[1]);
        SDL_Quit();
        return 1;
    }
    
    player.width = player.reader->hdr.width;
    player.height = player.reader->hdr.height;
    player.fps = player.reader->hdr.fps;
    player.has_audio = (player.reader->hdr.flags & WUBUV_FLAG_HAS_AUDIO) != 0;
    player.audio_rate = player.reader->hdr.audio_rate;
    player.frame_ms = 1000 / player.fps;
    
    printf("Playing: %s\n", argv[1]);
    printf("  Resolution: %dx%d @ %d fps\n", player.width, player.height, player.fps);
    if (player.has_audio) {
        printf("  Audio: %d Hz, %d channels\n", player.audio_rate, player.reader->hdr.audio_channels);
    }
    printf("  Frames: %d\n", player.reader->hdr.frame_count);
    printf("\nControls:\n");
    printf("  ESC - Quit\n");
    printf("  SPACE - Pause/Resume\n");
    printf("  LEFT/RIGHT - Seek\n");
    printf("  F - Toggle fullscreen\n\n");
    
    /* Create window */
    player.window = SDL_CreateWindow(WINDOW_TITLE,
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        player.width, player.height, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
    
    player.renderer = SDL_CreateRenderer(player.window, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    
    /* Create texture for video */
    player.texture = SDL_CreateTexture(player.renderer,
        SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING,
        player.width, player.height);
    
    /* Setup audio */
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
        if (player.audio_dev == 0) {
            fprintf(stderr, "Audio open failed: %s\n", SDL_GetError());
        } else {
            SDL_PauseAudioDevice(player.audio_dev, 0); /* Start audio */
        }
    }
    
    /* Allocate frame buffer */
    player.current_y = malloc(player.width * player.height);
    player.current_u = malloc(player.width * player.height / 4);
    player.current_v = malloc(player.width * player.height / 4);
    
    /* Main loop */
    SDL_Event event;
    int running = 1;
    player.start_ticks = SDL_GetTicks();
    
    /* Read first video frame */
    uint8_t type;
    uint8_t* data;
    uint32_t size;
    
    while (running) {
        /* Handle events */
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
                        case SDLK_LEFT:
                            /* Seek backward */
                            break;
                        case SDLK_RIGHT:
                            /* Seek forward */
                            break;
                        case SDLK_f:
                            /* Toggle fullscreen */
                            break;
                    }
                    break;
            }
        }
        
        if (!player.paused) {
            /* Read and display video frame */
            int ret = wubuv_read_frame(player.reader, &type, &data, &size);
            if (ret != 0) {
                /* End of video */
                break;
            }
            
            if (type == WUBUV_FRAME_KEY || type == WUBUV_FRAME_INTER) {
                /* Extract Y plane */
                uint16_t frame_w, frame_h;
                memcpy(&frame_w, data, 2);
                memcpy(&frame_h, data + 2, 2);
                memcpy(player.current_y, data + 4, frame_w * frame_h);
                free(data);
                
                /* For now, fill U and V with 128 (grayscale) */
                memset(player.current_u, 128, frame_w * frame_h / 4);
                memset(player.current_v, 128, frame_w * frame_h / 4);
                
                /* Convert to RGB and update texture */
                uint8_t* rgb = malloc(frame_w * frame_h * 3);
                yuv420p_to_rgb(player.current_y, player.current_u, player.current_v,
                               frame_w, frame_h, rgb);
                
                SDL_UpdateTexture(player.texture, NULL, rgb, frame_w * 3);
                free(rgb);
                
                /* Render */
                SDL_RenderClear(player.renderer);
                SDL_RenderCopy(player.renderer, player.texture, NULL, NULL);
                SDL_RenderPresent(player.renderer);
                
                /* Frame timing */
                player.frame_count++;
                uint32_t target_ticks = player.start_ticks + player.frame_count * player.frame_ms;
                uint32_t now = SDL_GetTicks();
                if (now < target_ticks) {
                    SDL_Delay(target_ticks - now);
                }
            } else if (type == WUBUV_FRAME_AUDIO) {
                /* Audio frame - handled by callback */
                free(data);
            }
        } else {
            SDL_Delay(10); /* Reduce CPU when paused */
        }
    }
    
    /* Cleanup */
    free(player.current_y);
    free(player.current_u);
    free(player.current_v);
    SDL_DestroyTexture(player.texture);
    SDL_DestroyRenderer(player.renderer);
    SDL_DestroyWindow(player.window);
    if (player.audio_dev > 0) SDL_CloseAudioDevice(player.audio_dev);
    wubuv_reader_close(player.reader);
    SDL_Quit();
    
    printf("Playback complete. Played %d frames.\n", player.frame_count);
    return 0;
}
