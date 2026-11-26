/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * CausalSplat: compile-time feature rasterization channels (per-Gaussian affinity dim).
 */

#ifndef CUDA_RASTERIZER_CONFIG_CAUSAL_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_CAUSAL_H_INCLUDED

#define CAUSAL_FEATURE_CHANNELS 32
#define NUM_CHANNELS CAUSAL_FEATURE_CHANNELS
#define BLOCK_X 16
#define BLOCK_Y 16

#endif
