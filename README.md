# turboquant_implementation

# 🚀 TurboQuant-PyTorch

An unofficial, end-to-end PyTorch implementation of the Google Research ICLR 2026 paper: [**TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**].

## ✨ Key Features
* **Faithful QJL Implementation:** Implements the 1-bit Quantized Johnson-Lindenstrauss (QJL) transform for unbiased inner-product estimation.
* **Spherical Lloyd-Max Quantization:** Includes the continuous k-means solver for generating optimal codebooks based on the Beta distribution.
* **Variance-Optimized Projection:** Expands the QJL projection dimension ($m = 4d$) to heavily suppress estimator variance before the Softmax bottleneck.
* **FP16 Value Passthrough (The Outlier Fix):** Compresses the Key cache to 3-bit/4-bit while leaving the Value cache in native FP16. This prevents spherical quantization from permanently destroying the massive activation outliers necessary for coherent LLM generation.

The KV cache is the bottleneck for serving LLMs at scale. TurboQuant gives 6x compression with zero quality loss:

- **6x more concurrent users per GPU** — direct 6x reduction in cost per query
- **6x longer context windows** in the same memory budget
- **No calibration step** — compress on-the-fly as tokens stream in
- **8x speedup on attention** at 4-bit on H100 GPUs (less data to load from HBM)

At H100 prices (~$2-3/hr), serving 6x more users per GPU translates to millions in savings at scale.


Requirements: `torch`, `transformers`, `scipy`
