import torch
from simple_knn._C import KNN

points = torch.rand(20, 3).cuda()
K = 3
dists = ((points.unsqueeze(0) - points.unsqueeze(1))**2).sum(-1)
best, indices = dists.topk(K + 1, dim=1, largest=False)
best, indices = best[:, 1:], indices[:, 1:]
best_out, indices_out = KNN(points)
print((best - best_out).abs().max(), (indices - indices_out).abs().max())