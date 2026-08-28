# WuBuCodec Makefile
# Pure C11 video codec — no external deps except zlib

CC = gcc
CFLAGS = -std=c11 -O3 -march=native -Iinclude -w
LDFLAGS = -lm -lpthread -lz -fopenmp

SRC = src
INC = include
OBJ = build
BIN = bin

# === Codec Core (C15 keystone) ===
CODEC_SRCS = $(SRC)/codec/wubu_encoder.c \
             $(SRC)/transform/wubu_transform.c \
             $(SRC)/transform/wubu_idct8x8.c \
             $(SRC)/transform/wubu_simd_dct.c \
             $(SRC)/entropy/wubu_expgolomb2.c \
             $(SRC)/entropy/wubu_cabac.c \
             $(SRC)/entropy/wubu_coeff.c \
             $(SRC)/entropy/wubu_zigrle.c \
             $(SRC)/entropy/wubu_trellis.c \
             $(SRC)/entropy/wubu_ncodec.c \
             $(SRC)/motion/wubu_motionest.c \
             $(SRC)/motion/wubu_mc2.c \
             $(SRC)/motion/wubu_subpel.c \
             $(SRC)/motion/wubu_mvprep.c \
             $(SRC)/motion/wubu_dpb.c \
             $(SRC)/intra/wubu_intra.c \
             $(SRC)/intra/wubu_angular.c \
             $(SRC)/intra/wubu_cclm.c \
             $(SRC)/filter/wubu_deblock.c \
             $(SRC)/filter/wubu_deblock2.c \
             $(SRC)/filter/wubu_sao.c \
             $(SRC)/filter/wubu_fgs.c \
             $(SRC)/temporal/wubu_bframe.c \
             $(SRC)/temporal/wubu_bframe2.c \
             $(SRC)/temporal/wubu_gop_opt.c \
             $(SRC)/temporal/wubu_lookahead.c \
             $(SRC)/temporal/wubu_fastmode.c \
             $(SRC)/rdo/wubu_rdo.c \
             $(SRC)/rdo/wubu_rdomode.c \
             $(SRC)/rdo/wubu_rd_curve.c \
             $(SRC)/container/wubu_nal.c \
             $(SRC)/container/wubuv.c \
             $(SRC)/container/wubuv2.c \
             $(SRC)/quality/wubu_bench_quality.c \
             $(SRC)/quality/wubu_ssim.c \
             $(SRC)/quality/wubu_satd.c \
             $(SRC)/neural/wubu_learned_codec.c \
             $(SRC)/neural/wubu_latent_codec.c \
             $(SRC)/quat/wubu_quat_codec.c \
             $(SRC)/quat/wubu_quat_rate.c \
             $(SRC)/quat/wubu_slerp_path.c \
             $(SRC)/quat/wubu_quaternion.c \
             $(SRC)/quat/wubu_quaternion_ops.c \
             $(SRC)/quat/wubu_so3.c \
             $(SRC)/quat/wubu_dualquat.c \
             $(SRC)/manifold/wubu_hyperbolic.c \
             $(SRC)/manifold/wubu_poincare_geom.c \
             $(SRC)/manifold/wubu_lorentz.c \
             $(SRC)/manifold/wubu_lorentz_poincare.c \
             $(SRC)/manifold/wubu_manifold.c \
             $(SRC)/manifold/wubu_manifold_ad.c \
             $(SRC)/manifold/wubu_parallel_transport.c \
             $(SRC)/manifold/wubu_tangent_flow.c \
             $(SRC)/manifold/wubu_flow_matching.c \
             $(SRC)/manifold/wubu_manifold_clip.c \
             $(SRC)/manifold/wubu_hclip.c \
             $(SRC)/manifold/wubu_lorentz_clip.c \
             $(SRC)/manifold/wubu_tree_embed.c \
             $(SRC)/manifold/wubu_hae.c \
             $(SRC)/manifold/wubu_hanalogy.c \
             $(SRC)/gaad/wubu_gaad_encoder.c \
             $(SRC)/audio/wubu_psychoacoustic.c \
             $(SRC)/audio/wubu_av_fidelity.c \
             $(SRC)/audio/wubu_stft.c \
             $(SRC)/audio/wubu_bands.c \
             $(SRC)/threed/wubu_360.c \
             $(SRC)/utils/wubu_utils.c \
             $(SRC)/utils/wubu_color.c \
             $(SRC)/utils/wubu_positional_encode.c \
             $(SRC)/utils/wubu_yuv420.c \
             $(SRC)/utils/wubu_simd.c \
             $(SRC)/utils/wubu_speed.c \
             $(SRC)/math/wubu_rep_theory.c \
             $(SRC)/math/wubu_anyon.c \
             $(SRC)/math/wubu_dasgupta.c \
             $(SRC)/math/wubu_kpp.c \
             $(SRC)/math/wubu_specnorm.c \
             $(SRC)/math/wubu_radam.c \
             $(SRC)/math/wubu_lrsched.c \
             $(SRC)/math/wubu_hamilton_encoder.c \
             $(SRC)/math/wubu_vhf_decoder.c \
             $(SRC)/math/wubu_vhf_audio.c \
             $(SRC)/math/wubu_canvas.c \
             $(SRC)/math/wubu_nested_encoder.c \
             $(SRC)/math/wubu_scene.c \
             $(SRC)/math/wubu_q_controller.c \
             $(SRC)/math/wubu_loss.c \
             $(SRC)/math/wubu_ratecontrol.c \
             $(SRC)/math/wubu_frame_parallel.c \
             $(SRC)/math/wubu_tile_parallel.c

CODEC_OBJS = $(patsubst $(SRC)/%.c,$(OBJ)/%.o,$(CODEC_SRCS))

# === Test targets ===
TEST_SRCS = $(wildcard $(SRC)/../tests/*.c)
TEST_BINS = $(patsubst $(SRC)/../tests/%.c,$(BIN)/%,$(TEST_SRCS))

# === Targets ===

all: codec tests

codec: $(BIN)/wubucodec

$(BIN)/wubucodec: wubucodec.c $(CODEC_OBJS)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $< $(CODEC_OBJS) -o $@ $(LDFLAGS)

$(OBJ)/%.o: $(SRC)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Tests
tests: $(TEST_BINS)
	@echo "=== Running all tests ==="
	@for t in $(TEST_BINS); do echo "--- $$t ---"; ./$$t 2>&1 | tail -3; done

$(BIN)/%: $(SRC)/../tests/%.c $(CODEC_OBJS)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $< $(CODEC_OBJS) -o $@ $(LDFLAGS)

# Specific category tests
test-c15: $(BIN)/test_wubu_encoder
	./$(BIN)/test_wubu_encoder

test-c10: $(BIN)/test_wubu_trellis
	./$(BIN)/test_wubu_trellis

test-c9: $(BIN)/test_wubu_transform
	./$(BIN)/test_wubu_transform

test-c12: $(BIN)/test_wubu_deblock2
	./$(BIN)/test_wubu_deblock2

test-c1: $(BIN)/test_wubu_cabac
	./$(BIN)/test_wubu_cabac

test-gap: $(BIN)/wubu_gap_closure
	./$(BIN)/wubu_gap_closure

test-c4: $(BIN)/test_wubu_coeff
	./$(BIN)/test_wubu_coeff

test-speed: $(BIN)/wubu_speed_bench
	./$(BIN)/wubu_speed_bench

test-preset: $(BIN)/wubu_preset_bench
	./$(BIN)/wubu_preset_bench

test-1080p: $(BIN)/wubu_1080p_bench
	./$(BIN)/wubu_1080p_bench

test-8k: $(BIN)/wubu_8k_bench
	./$(BIN)/wubu_8k_bench

test-tile: $(BIN)/wubu_tile_bench
	./$(BIN)/wubu_tile_bench

# Clean
clean:
	rm -rf $(OBJ) $(BIN)

.PHONY: all codec tests clean test-c15 test-c10 test-c9 test-c12 test-c1 test-c4
