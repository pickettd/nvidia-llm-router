# Edge Deployment: Raspberry Pi 5 & NVIDIA Jetson

Notes on deploying the LLM Router to low-power edge devices.

## Architecture Recap

The router pipeline has three stages, each with different compute requirements:

| Stage | Backend | Model Size | Compute |
|-------|---------|-----------|---------|
| Preprocessing (tokenization) | Python | ~500MB (DeBERTa-v2 tokenizer downloaded from HuggingFace at startup) | CPU only |
| Classification (DeBERTa-v2) | PyTorch / libtorch | 702MB `.pt` file | GPU or CPU |
| Postprocessing (softmax → one-hot) | Python | Negligible | CPU only |

The classification model is a DeBERTa-v2 transformer (~125M params). Total memory for one router policy: ~1.5GB (model weights + tokenizer + runtime overhead). Both `task_router` and `complexity_router` share the same architecture, so loading both requires ~3GB.

## NVIDIA Jetson (Recommended for Edge)

Jetson devices (Orin Nano, Orin NX, AGX Orin) are the more natural fit since they have NVIDIA GPUs and official Triton support.

### What works out of the box

- NVIDIA provides Jetson-specific Triton containers: `nvcr.io/nvidia/tritonserver:<version>-jetpack-py3`
- The PyTorch libtorch backend runs on Jetson GPUs
- The existing `KIND_GPU` model configs work as-is
- The Router Controller (Rust binary) needs to be cross-compiled for `aarch64`

### Jetson considerations

- **Memory**: Jetson shares RAM between CPU and GPU. The Orin Nano (8GB) is tight — a single router policy (~1.5GB) plus Triton overhead plus the OS would leave little headroom. The Orin NX (16GB) or AGX Orin (32/64GB) are better choices.
- **Triton container version**: Check the [Jetson container compatibility matrix](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver) for your JetPack version. Blackwell-specific containers are not needed since Jetson uses its own GPU architecture (Ampere on Orin).
- **Router Controller**: The Rust binary in `router-controller.dockerfile` builds for x86_64. For Jetson (aarch64), either cross-compile or build natively on the device:
  ```bash
  # On Jetson or via cross-compilation
  cd src/router-controller
  cargo build --release --target aarch64-unknown-linux-gnu
  ```
- **Downstream LLMs**: Ollama has ARM64 builds and runs on Jetson. Small models like `qwen3:0.6b` and `gemma3:1b` run well on Jetson GPUs.

## Raspberry Pi 5 (CPU-Only)

The Pi 5 has no NVIDIA GPU, so the entire pipeline must run on CPU. This is feasible but requires model conversion and will have higher classification latency.

### Model conversion (GPU → CPU)

The pre-trained `.pt` files were traced on GPU and contain `cuda:0` tensors. Loading them with `KIND_CPU` in Triton fails with a device mismatch error. The models must be re-saved with CPU tensors:

```python
import torch

# Convert task_router
model = torch.jit.load("routers/task_router/1/model.pt", map_location="cpu")
torch.jit.save(model, "routers/task_router/1/model.pt")

# Convert complexity_router
model = torch.jit.load("routers/complexity_router/1/model.pt", map_location="cpu")
torch.jit.save(model, "routers/complexity_router/1/model.pt")
```

### Triton config changes

Change `KIND_GPU` to `KIND_CPU` in both model configs:

```
# routers/task_router/config.pbtxt
# routers/complexity_router/config.pbtxt

instance_group [
    {
    kind: KIND_CPU
    count: 1
    }
]
```

CPU override configs are already available in `routers-cpu-override/` for reference.

### Triton on ARM64 / Pi 5

Triton does not have official Raspberry Pi builds. Options:

1. **Build Triton from source for aarch64** — Triton's [build instructions](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md) support ARM. Only the Python backend and PyTorch backend are needed. This is non-trivial.

2. **Replace Triton with a lightweight Python inference server** — Since the Pi only needs CPU inference on a single TorchScript model, a minimal Flask/FastAPI server that loads the model with `torch.jit.load()` and exposes the same `/v2/models/.../infer` API would be simpler. The pre/post processing Python code from `routers/preprocessing_*/1/model.py` and `routers/postprocessing_*/1/model.py` can be reused directly.

3. **Convert to ONNX and use ONNX Runtime** — ONNX Runtime has excellent ARM64/CPU support and would likely be faster than PyTorch on CPU. The DeBERTa-v2 model can be exported:
   ```python
   import torch
   model = torch.jit.load("routers/task_router/1/model.pt", map_location="cpu")
   dummy_ids = torch.zeros(1, 512, dtype=torch.long)
   dummy_mask = torch.zeros(1, 512, dtype=torch.long)
   torch.onnx.export(model, (dummy_ids, dummy_mask), "task_router.onnx",
                      input_names=["input_ids", "attention_mask"],
                      output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}})
   ```

### Pi 5 performance expectations

- **RAM**: 16GB Pi 5 can hold both router models (~3GB) plus Ollama with small models. Tight but workable.
- **Classification latency**: DeBERTa-v2 inference on a Cortex-A76 (Pi 5 CPU) will likely take 1-5 seconds per prompt. This is acceptable for a routing decision since the downstream LLM generation takes much longer.
- **Router Controller**: The Rust binary compiles for `aarch64-unknown-linux-gnu` and is lightweight.
- **Downstream LLMs**: Ollama supports ARM64. Models like `qwen3:0.6b` (~600MB) run on Pi 5 but generation is slow (~5-15 tokens/sec). The router adds value by steering complex prompts to a remote/larger model and only running simple prompts locally.

## Summary

| | Jetson (Orin NX/AGX) | Raspberry Pi 5 (16GB) |
|---|---|---|
| GPU inference | Yes (native) | No |
| Triton support | Official Jetson containers | No official build; use alternative |
| Model conversion needed | No | Yes (`map_location="cpu"`) |
| Classification latency | ~50-100ms | ~1-5s |
| Recommended approach | Triton + Ollama on device | Lightweight Python server + ONNX Runtime |
| Best for | Self-contained edge router | Low-cost routing with remote LLM fallback |
