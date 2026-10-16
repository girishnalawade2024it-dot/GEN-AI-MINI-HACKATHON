#!/usr/bin/env python3
"""
ScamShield: Standalone ONNX Export and INT8 Quantization Script

Loads the trained PyTorch checkpoint from ./output/best_model, exports to ONNX,
quantizes dynamically to INT8, and copies directly to Android app assets.
"""

import os
import sys
import shutil

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def export_and_quantize(
    model_dir: str = "./output/best_model",
    output_dir: str = "./output/android_assets",
    max_length: int = 64,
):
    print(f"[*] Loading fine-tuned model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model.cpu()

    os.makedirs(output_dir, exist_ok=True)

    # 1. Export Vocab for Android Native Tokenizer
    vocab_path = os.path.join(output_dir, "vocab.txt")
    print(f"[*] Exporting tokenizer vocabulary to {vocab_path}...")
    vocab_dict = tokenizer.get_vocab()
    sorted_tokens = sorted(vocab_dict.items(), key=lambda x: x[1])
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, _ in sorted_tokens:
            f.write(token + "\n")
    print(f"[OK] Exported {len(sorted_tokens)} vocabulary tokens.")

    # 2. Export to ONNX (Raw FP32)
    onnx_raw_path = os.path.join(output_dir, "scam_model_raw.onnx")
    print(f"[*] Exporting to ONNX: {onnx_raw_path}...")
    dummy_text = "Your Chase bank account is suspended. Verify now: http://phish.com"
    inputs = tokenizer(
        dummy_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        onnx_raw_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(onnx_raw_path)
    onnx.checker.check_model(onnx_model)
    raw_size_mb = os.path.getsize(onnx_raw_path) / (1024 * 1024)
    print(f"[OK] Raw ONNX Model successfully created: {raw_size_mb:.2f} MB")

    # 3. Dynamic INT8 Quantization (<35MB constraint)
    onnx_quant_path = os.path.join(output_dir, "scam_model_quantized.onnx")
    print(f"[*] Quantizing model dynamically to INT8: {onnx_quant_path}...")
    quantize_dynamic(
        model_input=onnx_raw_path,
        model_output=onnx_quant_path,
        op_types_to_quantize=["MatMul", "Gemm", "Gather"],
        weight_type=QuantType.QInt8,
    )

    quant_size_mb = os.path.getsize(onnx_quant_path) / (1024 * 1024)
    print(f"[OK] Quantized ONNX Model Size: {quant_size_mb:.2f} MB (Target: < 35.0 MB)")

    # 4. Copy to Android Assets
    android_assets_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "android", "app", "src", "main", "assets")
    )
    if os.path.exists(android_assets_dir):
        print(f"[*] Deploying artifacts to Android assets: {android_assets_dir}...")
        shutil.copy2(onnx_quant_path, os.path.join(android_assets_dir, "scam_model_quantized.onnx"))
        shutil.copy2(vocab_path, os.path.join(android_assets_dir, "vocab.txt"))
        print("[OK] Successfully copied scam_model_quantized.onnx and vocab.txt to Android assets!")

    # Clean up raw FP32 ONNX to save disk space
    if os.path.exists(onnx_raw_path):
        os.remove(onnx_raw_path)

    print("\n=======================================================")
    print("[DONE] ONNX Export and INT8 Quantization Completed!")
    print(f"[*] Quantized Model: {onnx_quant_path} ({quant_size_mb:.2f} MB)")
    print(f"[*] Vocab:           {vocab_path}")
    print("=======================================================\n")


if __name__ == "__main__":
    export_and_quantize()
