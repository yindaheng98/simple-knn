/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "spatial.h"
#include "simple_knn.h"

std::tuple<torch::Tensor, torch::Tensor>
KNN(const torch::Tensor& points)
{
  const int P = points.size(0);

  auto int_opts = points.options().dtype(torch::kInt32);
  torch::Tensor idx = torch::full({P, 3}, -1, int_opts);
  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor dist = torch::full({P, 3}, 0.0, float_opts);
  
  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), idx.contiguous().data<int>(), dist.contiguous().data<float>());

  return std::make_tuple(dist, idx);
}