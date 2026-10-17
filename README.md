# ClickSmith Live — models

Model files the ClickSmith Live app downloads on first use. Each one is published as a release asset with its SHA-256; the app checks size and checksum before using it.

## SCNet Tran Small — release `scnet-tran-v1`

| | |
|---|---|
| File | `scnet_tran_core_fp16.onnx` (26,957,934 bytes) |
| SHA-256 | `46de9983af079ed349887ce7b363c36fc988539500f9f63c3a89c7886faa4ee6` |
| What it does | 4-stem music source separation: drums, bass, other, vocals (44.1 kHz stereo) |
| Original | `model_scnet_tran_sdr_8.9272.ckpt` from [ZFTurbo/Music-Source-Separation-Training, release v1.0.14](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.14) |
| Weights by | **ZFTurbo** (Roman Solovyev) — released under the MIT License |
| Architecture | SCNet — Tong et al., *SCNet: Sparse Compression Network for Music Source Separation* (2024), transformer variant from the same repository |
| Training data | According to the original release: MUSDB18 only |

**What we changed:** only the format. The network's core (encoder, separation network, decoder) is exported to ONNX (opset 17), with the STFT and iSTFT left outside the graph, the time-axis FFT inside the separation network replaced by an exact DFT matrix multiply (fixed 476 frames per 11 s chunk), and the weights stored as fp16. Output matches the original PyTorch model within 1.5e-4 (waveform), and both score the same.

**To use it:** input `spec` `[batch, 4, 2049, 476]` float32 = the STFT of an 11 s chunk (485,100 samples, padded to an odd frame count as upstream does; n_fft 4096, hop 1024, rectangular window, centred with reflect padding, normalised), channels `[L.re, L.im, R.re, R.im]`. Output `out` `[batch, 16, 2049, 476]` = the same four channels for drums, bass, other, vocals.

## SCNet Tran Small for Core ML — release `scnet-tran-coreml-v1`

The same network, compiled for Apple's Core ML so it runs on a Mac's GPU (about 8–10× faster than the ONNX file on the CPU on an M1 Pro).

| | |
|---|---|
| File | `scnet_tran_fp32.mlmodelc.zip` (39,220,331 bytes) — a compiled `scnet_tran_fp32.mlmodelc` folder |
| SHA-256 | `e0993fbe7c3068673088751fbcedcf6194ad08ef7634f14236a883798154c443` |
| Original | `model_scnet_tran_sdr_8.9272.ckpt`, as above |
| Weights by | **ZFTurbo** (Roman Solovyev) — released under the MIT License |
| Needs | macOS 13 or later; runs on CPU + GPU (`MLComputeUnits.cpuAndGPU`) |

**What we changed:** only the format. The same core as the ONNX file (STFT and iSTFT outside, DFT matrix multiply in place of the time-axis FFT), converted from the PyTorch checkpoint with coremltools 9.0 as an fp32 ML Program (deployment target macOS 13) and compiled with `xcrun coremlcompiler`. Output matches the original PyTorch model to 108 dB SNR, and both score the same.

**To use it:** input `spec` `[1, 4, 2049, 476]` float32 and output `out` `[1, 16, 2049, 476]` float32, laid out as for the ONNX file (batch fixed at 1).

## Licence

MIT — see [LICENSE](LICENSE). Please credit **ZFTurbo** for the weights. Provided as-is, without warranty.
