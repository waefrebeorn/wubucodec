# WuBuCodec — Complete Build System
# Builds: library, tools, player, and benchmarks

CC = gcc
CFLAGS = -std=c11 -O3 -march=native -Iinclude -Wall -Wno-unused-function
LDFLAGS = -lm -lpthread -lz

SRC = src
INC = include
OBJ = build
BIN = bin

# Core library sources
LIB_SRCS = \
    $(SRC)/transform/wubu_transform.c \
    $(SRC)/motion/wubu_motionest.c \
    $(SRC)/audio/wubu_stft.c \
    $(SRC)/audio/wubu_kodak.c \
    $(SRC)/entropy/wubu_entropy.c \
    $(SRC)/container/wubuv.c

LIB_OBJS = $(patsubst $(SRC)/%.c,$(OBJ)/%.o,$(LIB_SRCS))

# Targets
all: dirs $(BIN)/libwubuv.a $(BIN)/wubuv_encode $(BIN)/wubuv_decode $(BIN)/wubuv_play $(BIN)/wubuv_info $(BIN)/wubuv_verify

dirs:
	@mkdir -p $(BIN) $(OBJ)/transform $(OBJ)/motion $(OBJ)/audio $(OBJ)/container $(OBJ)/entropy $(OBJ)/tools

# Static library
$(BIN)/libwubuv.a: $(LIB_OBJS)
	ar rcs $@ $^

# Library object files
$(OBJ)/%.o: $(SRC)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Tools
$(BIN)/wubuv_encode: tools/wubuv_encode.c $(LIB_OBJS)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) -o $@ $(LDFLAGS)

$(BIN)/wubuv_decode: tools/wubuv_decode.c $(LIB_OBJS)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) -o $@ $(LDFLAGS)

$(BIN)/wubuv_play: tools/wubuv_player.c $(LIB_OBJS)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) -o $@ $(LDFLAGS) -lSDL2

$(BIN)/wubuv_info: tools/wubuv_info.c $(LIB_OBJS)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) -o $@ $(LDFLAGS)

$(BIN)/wubuv_verify: tools/wubuv_verify.c $(LIB_OBJS)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) -o $@ $(LDFLAGS)

# Clean
clean:
	rm -rf $(OBJ) $(BIN)

# Install
install: all
	cp $(BIN)/wubuv_encode /usr/local/bin/
	cp $(BIN)/wubuv_decode /usr/local/bin/
	cp $(BIN)/wubuv_play /usr/local/bin/
	cp $(BIN)/wubuv_info /usr/local/bin/
	cp $(BIN)/wubuv_verify /usr/local/bin/

.PHONY: all dirs clean install
