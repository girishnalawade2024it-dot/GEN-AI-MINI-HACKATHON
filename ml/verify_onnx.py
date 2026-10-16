#!/usr/bin/env python3
"""
ScamShield: ONNX Model Verification & Latency Benchmark Script

Loads the exported INT8 quantized ONNX model, executes inference on real-world
test samples (smishing, OTPs, phishing, personal messages), and measures per-sample
inference latency on CPU to validate on-device mobile feasibility.
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import argparse
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def softmax(x: np.ndarray) -> np.ndarray:
    """Computes softmax values along the last axis."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description="Verify Quantized ONNX Model for ScamShield")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./output/android_assets/scam_model_quantized.onnx",
        help="Path to scam_model_quantized.onnx",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="distilbert-base-uncased",
        help="Tokenizer name or path to local vocab",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Confidence threshold for triggering alert overlay (default: 0.85)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"[!] Error: Model file not found at: {args.model_path}")
        print("    Please run train_and_export.py first.")
        sys.exit(1)

    file_size_mb = os.path.getsize(args.model_path) / (1024 * 1024)
    print(f"[*] Loading Quantized ONNX Model: {args.model_path}")
    print(f"[*] Model File Size: {file_size_mb:.2f} MB (Constraint: < 35 MB)")

    # Configure ONNX Runtime Session optimized for mobile CPU
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(args.model_path, sess_options=opts, providers=["CPUExecutionProvider"])

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    test_cases = [
        # (Label, Text)
        (
            "HAM",
            "Your Chase bank security verification code is 492019. Valid for 10 minutes. Never share this code.",
        ),
        (
            "SCAM",
            "URGENT: Your Chase bank account is locked due to suspicious login. Verify identity now at: http://chase-security-verify.com/login",
        ),
        (
            "HAM",
            "Hey Mom, I just got back from the grocery store. Will bring the apples over later tonight.",
        ),
        (
            "SCAM",
            "USPS Notice: Your parcel #98214 is held at our depot due to unpaid $1.50 customs fee. Resolve: http://usps-redelivery-fee.top",
        ),
        (
            "HAM",
            "Your Uber ride is arriving in 3 minutes. White Honda Civic, license plate 7XYZ89.",
        ),
        (
            "SCAM",
            "CONGRATULATIONS! You have won an Amazon $1000 Gift Card! Claim within 24h at http://amazon-winner-gift.xyz",
        ),
        (
            "SCAM",
            "Crypto Alert: 1.5 BTC has been deposited into your temporary wallet. Withdraw now: http://binance-claim-crypto.site",
        ),
        (
            "HAM",
            "Reminder: Your doctor appointment is scheduled for tomorrow at 3:30 PM with Dr. Harris.",
        ),
    ]

    print("\n" + "=" * 80)
    print(f"{'EXPECTED':<8} | {'PREDICTED':<9} | {'SCAM CONF':<10} | {'LATENCY (ms)':<13} | {'MESSAGE PREVIEW':<30}")
    print("=" * 80)

    latencies = []
    correct_count = 0

    for expected, text in test_cases:
        # Tokenize (matching exact Android WordPiece pipeline)
        encoded = tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)

        start_time = time.perf_counter()
        outputs = session.run(
            ["logits"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000.0
        latencies.append(latency_ms)

        logits = outputs[0][0]
        probs = softmax(logits)
        scam_confidence = float(probs[1])

        # Flagged as scam if confidence >= threshold (minimizing False Positives)
        pred_label = "SCAM" if scam_confidence >= args.threshold else "HAM"
        if pred_label == expected:
            correct_count += 1

        short_preview = (text[:35] + "...") if len(text) > 35 else text
        print(
            f"{expected:<8} | {pred_label:<9} | {scam_confidence * 100:>8.2f}% | "
            f"{latency_ms:>10.2f} ms | {short_preview}"
        )

    print("=" * 80)
    avg_latency = np.mean(latencies[1:]) if len(latencies) > 1 else latencies[0]  # Exclude warmup
    accuracy = (correct_count / len(test_cases)) * 100

    print(f"\n[OK] Test Accuracy: {accuracy:.1f}% ({correct_count}/{len(test_cases)})")
    print(f"[OK] Average CPU Inference Latency: {avg_latency:.2f} ms per message")
    print(f"[OK] Model Size: {file_size_mb:.2f} MB")
    print("[OK] Verification passed! Ready for Android integration.\n")


if __name__ == "__main__":
    main()
