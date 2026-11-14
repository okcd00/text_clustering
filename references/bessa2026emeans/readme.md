# e-means: outperforming k-means (in one-tenth the epochs)

Citation key: `bessa2026emeans`

- Bessa et al., arXiv preprint, 2026-11-06. [Version reviewed](https://arxiv.org/abs/2611.09501v1) · [Full text](https://arxiv.org/html/2611.09501v1) · [Faiss v1.16.0 implementation and CPU instructions](https://github.com/facebookresearch/faiss/blob/v1.16.0/contrib/e_means/README.md).
- Scope: **fixed-K batch vector clustering / throughput**, not unknown-K text topic discovery, online novelty detection, or an intent classifier. CPU implementation uses Faiss and NumPy. No local execution or package installation was performed.

## What the paper measures

e-means updates centroids with exponential moving averages over mini-batches. Section D.3, Table 11 reports time to reach the held-out Euclidean distortion of Lloyd at 25 epochs, not time to match intent labels. On 64 physical CPU cores with float32, the authors report 15.7 s vs 178.0 s on AG-News (0.76M embeddings), and 24.6 s vs 252.9 s on arXiv (1.34M embeddings); both use K=16,384. These are VIBE benchmark settings, not the task's category count. The results demonstrate a time-to-distortion tradeoff at those operating points; they do not establish better semantic ARI, lower 512-row latency, or a drop-in win for text clustering.

## Fair comparison if throughput becomes a bottleneck

1. Put it in a separate fixed-K scalability track. Keep the same frozen embedding matrix, dtype, K, metric, hardware/thread budget and seed list for e-means, Lloyd and MiniBatchKMeans. Match initialization where supported; otherwise disclose differences and report results across seeds. Use more than one input size, including a genuinely large workload; don't substitute duplicated texts to create scale.
2. Report wall time and peak memory excluding embedding generation, plus distortion and, where labels exist, ARI/NMI. Show both fixed-time and time-to-matched-distortion comparisons. Report fitting and final assignment times separately, and include both in end-to-end throughput.
3. Faiss `train(X)` returns labels from the last epoch while centroids were still moving. Use `assign(X)` for exact final-centroid labels in the matched evaluation and include the extra pass in timing. If reporting native last-pass labels, mark them separately.
4. Keep the current small semantic pilot unchanged. Even if e-means wins at scale, a fixed-K method does not resolve unknown K or whether a sample should be left unassigned.

Status: primary-source evidence only; no survey benchmark result.
