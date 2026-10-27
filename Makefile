# WuBuCodec — Build with arithmetic coding
CC = gcc
CFLAGS = -std=c11 -O3 -march=native -Iinclude -Wall -Wno-unused-function
LDFLAGS = -lm -lpthread -lz

SRC = src
OBJ = build
BIN = bin

LIB_SRCS = \
    $(SRC)/transform/wubu_transform.c \
    $(SRC)/motion/wubu_motionest.c \
    $(SRC)/audio/wubu_stft.c \
    $(SRC)/audio/wubu_kodak.c \
    $(SRC)/entropy/wubu_entropy.c \
    $(SRC)/entropy/wubu_cabac.c \
    $(SRC)/entropy/wubu_ac.c \
    $(SRC)/container/wubuv.c

LIB_OBJS = $(patsubst $(SRC)/%.c,$(OBJ)/%.o,$(LIB_SRCS))

all: dirs $(BIN)/wubuv_encode $(BIN)/wubuv_decode $(BIN)/wubuv_play $(BIN)/wubuv_info $(BIN)/wubuv_verify $(BIN)/wubuv_encode_v7 $(BIN)/wubuv_encode_v8 $(BIN)/wubu_latent_codec $(BIN)/wubu_canvas_engine

dirs:
	@mkdir -p $(BIN) $(OBJ)/transform $(OBJ)/motion $(OBJ)/audio $(OBJ)/entropy $(OBJ)/container

$(OBJ)/%.o: $(SRC)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN)/wubuv_encode: tools/wubuv_encode.c $(LIB_OBJS)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) -o $@ $(LDFLAGS)

$(BIN)/wubuv_encode_v7: tools/wubuv_encode_v7.c $(LIB_OBJS)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) -o $@ $(LDFLAGS)

$(BIN)/wubuv_encode_v8: tools/wubuv_encode_v8.c $(LIB_OBJS)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) -o $@ $(LDFLAGS)

$(BIN)/wubu_latent_codec: tools/wubu_latent_codec.c $(LIB_OBJS)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) -o $@ $(LDFLAGS)

$(BIN)/wubuv_decode: tools/wubuv_decode.c $(LIB_OBJS)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) -o $@ $(LDFLAGS)

$(BIN)/wubuv_play: tools/wubuv_player.c $(LIB_OBJS)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) -o $@ $(LDFLAGS) -lSDL2

$(BIN)/wubuv_info: tools/wubuv_info.c $(LIB_OBJS)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) -o $@ $(LDFLAGS)

$(BIN)/wubuv_verify: tools/wubuv_verify.c $(LIB_OBJS)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) -o $@ $(LDFLAGS)

clean:
	rm -rf $(OBJ) $(BIN)

.PHONY: all dirs clean
