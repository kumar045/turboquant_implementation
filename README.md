# turboquant_implementation

# 🚀 TurboQuant-PyTorch

An unofficial, end-to-end PyTorch implementation of the Google Research ICLR 2026 paper: [**TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**].

## ✨ Key Features
* **Faithful QJL Implementation:** Implements the 1-bit Quantized Johnson-Lindenstrauss (QJL) transform for unbiased inner-product estimation.
* **Spherical Lloyd-Max Quantization:** Includes the continuous k-means solver for generating optimal codebooks based on the Beta distribution.
* **Variance-Optimized Projection:** Expands the QJL projection dimension ($m = 4d$) to heavily suppress estimator variance before the Softmax bottleneck.
* **FP16 Value Passthrough (The Outlier Fix):** Compresses the Key cache to 3-bit/4-bit while leaving the Value cache in native FP16. This prevents spherical quantization from permanently destroying the massive activation outliers necessary for coherent LLM generation.

---

Requirements: `torch`, `transformers`, `scipy`
