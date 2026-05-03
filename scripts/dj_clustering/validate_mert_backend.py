"""
PURPOSE: Standalone MERT backend validation script. Run inside Apptainer on an a100 GPU
         node to confirm that the SIF, transformers, and MERT-v1-330M all work correctly
         before running full feature extraction.

CHANGELOG:
  D2.2 - Initial implementation.
"""

from __future__ import annotations

import sys


def main() -> int:
    print("=" * 70)
    print("[MERT BACKEND] Validation starting")
    print("=" * 70)

    # Step 1: imports
    try:
        import torch
        print(f"[MERT BACKEND] torch: {torch.__version__}")
        print(f"[MERT BACKEND] cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[MERT BACKEND] GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as exc:
        print(f"[MERT BACKEND] FAIL: torch import error: {exc}")
        return 1

    try:
        import transformers
        print(f"[MERT BACKEND] transformers: {transformers.__version__}")
    except ImportError as exc:
        print(f"[MERT BACKEND] FAIL: transformers import error: {exc}")
        return 1

    try:
        import numpy as np
    except ImportError as exc:
        print(f"[MERT BACKEND] FAIL: numpy import error: {exc}")
        return 1

    # Step 2: load model and processor
    model_name = "m-a-p/MERT-v1-330M"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MERT BACKEND] Loading model: {model_name} on {device}")

    try:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model = model.to(device)
        model.eval()
        print("[MERT BACKEND] Model and processor loaded successfully")
    except Exception as exc:
        print(f"[MERT BACKEND] FAIL: model load error: {type(exc).__name__}: {exc}")
        return 1

    # Step 3: synthetic forward pass (1s @ 24kHz)
    try:
        audio = np.zeros(24000, dtype=np.float32)
        inputs = processor(
            audio,
            sampling_rate=24000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(device)

        with torch.no_grad():
            outputs = model(input_values, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        last = hidden_states[-1].squeeze(0).mean(dim=0)
        shape = tuple(last.shape)
        print(f"[MERT BACKEND] Synthetic forward pass: embedding shape {shape}")
        print(f"[MERT BACKEND] Number of hidden state layers: {len(hidden_states)}")
    except Exception as exc:
        print(f"[MERT BACKEND] FAIL: forward pass error: {type(exc).__name__}: {exc}")
        return 1

    print("=" * 70)
    print("[MERT BACKEND] PASS")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
