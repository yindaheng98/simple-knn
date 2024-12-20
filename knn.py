import torch
from simple_knn import knn_batch, knn_kernel

points = torch.rand(10000, 3).cuda()
K = 3
dists = ((points.unsqueeze(0) - points.unsqueeze(1))**2).sum(-1).sqrt()
best, indices = dists.topk(K + 1, dim=1, largest=False)
best, indices = best[:, 1:], indices[:, 1:]
indices_kernel, best_kernel = knn_kernel(points, K)
print((best - best_kernel).abs().max(), (indices - indices_kernel).abs().max())
indices_batch, best_batch = knn_batch(points, K, batch=8)
print((best - best_batch).abs().max(), (indices - indices_batch).abs().max())
