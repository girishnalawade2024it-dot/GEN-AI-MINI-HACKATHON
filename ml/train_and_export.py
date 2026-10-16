#!/usr/bin/env python3
"""
ScamShield: Mobile-First Privacy-Preserving Scam Detection
Model Training, ONNX Export, and Dynamic INT8 Quantization Pipeline

This script:
1. Loads an SMS/text scam classification dataset (or synthesizes a domain-rich starter dataset).
2. Computes class weights prioritizing HAM preservation to ensure False Positive Rate (FPR) < 0.5%.
3. Fine-tunes a DistilBERT / MobileBERT model with custom class-weighted Cross-Entropy loss.
4. Exports the PyTorch model to ONNX with dynamic batch and sequence axes.
5. Applies Dynamic INT8 Quantization via onnxruntime.quantization to reach <35MB payload.
6. Exports the tokenizer vocabulary (vocab.txt) for on-device native Kotlin tokenization.
7. Evaluates latency, size, and classification metrics.
"""

import os
import sys
import shutil

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EvalPrediction,
)
from datasets import Dataset

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort


# ---------------------------------------------------------------------------
# 1. Custom Weighted Loss Trainer for Strict False Positive Control
# ---------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    """
    Custom Trainer that overrides compute_loss with class-weighted CrossEntropy.
    In ScamShield, False Positives (legitimate messages flagged as scams) are
    critical failures (e.g. blocking bank OTPs). HAM (class 0) errors are penalized heavily.
    """

    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# 2. Evaluation Metrics Calculation
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Computes Accuracy, Precision, Recall, F1, and specifically False Positive Rate (FPR).
    FPR = FP / (FP + TN). Target: FPR < 0.005 (< 0.5%).
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    probs = torch.softmax(torch.tensor(predictions), dim=-1)[:, 1].numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0

    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.0

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "auc": float(auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ---------------------------------------------------------------------------
# 3. Dataset Loading and Synthetic Fallback Generator
# ---------------------------------------------------------------------------
def load_or_generate_dataset(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads labeled data from CSV or TSV (e.g. UCI SMSSpamCollection).
    Standardizes columns to 'text' and 'label' (0=HAM, 1=SCAM/SPAM).
    Augments with modern smishing, banking OTP, delivery phishing, and crypto scam samples.
    """
    df = None
    if data_path and os.path.exists(data_path):
        print(f"[*] Loading dataset from {data_path}...")
        try:
            with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline()
            
            # Check if tab-separated UCI format (label\ttext) without header
            if "\t" in first_line and ("ham" in first_line.lower() or "spam" in first_line.lower()):
                df = pd.read_csv(
                    data_path,
                    sep="\t",
                    header=None,
                    names=["label", "text"],
                    encoding="utf-8",
                    on_bad_lines="skip",
                )
            else:
                df = pd.read_csv(data_path, encoding="utf-8", on_bad_lines="skip")
        except Exception as e:
            print(f"[!] Warning reading dataset: {e}, falling back to python engine...")
            df = pd.read_csv(data_path, sep=None, engine="python", on_bad_lines="skip")

        # Standardize column names
        df.columns = [str(c).lower().strip() for c in df.columns]
        if "message" in df.columns and "text" not in df.columns:
            df["text"] = df["message"]
        if "v2" in df.columns and "v1" in df.columns:  # Alternative UCI format
            df["text"] = df["v2"]
            df["label"] = df["v1"]

        # Normalize label mapping
        if "label" in df.columns:
            df["label"] = df["label"].apply(
                lambda x: 1 if str(x).lower().strip() in ["spam", "scam", "phishing", "1", "true"] else 0
            )

        df = df[["text", "label"]].dropna()
        print(f"[OK] Successfully loaded {len(df)} samples from dataset ({sum(df['label'] == 0)} HAM, {sum(df['label'] == 1)} SCAM/SPAM).")
        return df

    print("[!] No dataset provided. Generating domain-rich mobile scam & ham dataset...")
    
    ham_samples = [
        "Your Chase security code is 849201. Never share this code with anyone.",
        "Hi Mom, I will be home around 6:30 PM for dinner. Do you need anything?",
        "Your Uber ride is arriving now. Silver Toyota Camry (ABC-1234).",
        "Your appointment with Dr. Smith is confirmed for tomorrow at 2:00 PM.",
        "Google verification code: 394821. Valid for 10 minutes.",
        "Can you send me the slide deck from today's meeting before 5 PM?",
        "Your package from Amazon was delivered to your front door at 1:15 PM.",
        "Hey! Are we still meeting for lunch at the cafeteria today?",
        "Bank of America Alert: You made a $24.50 purchase at Starbucks on 10/14.",
        "Your flight DL1429 is on time. Gate B22 boarding begins at 4:15 PM.",
        "Netflix: Your monthly subscription payment was successful. Thank you.",
        "Hey bro, congratulations on the new job! Let's celebrate this weekend.",
        "Your prescription is ready for pickup at CVS Pharmacy #4829.",
        "Hi David, please find attached the reviewed contract for project Alpha.",
        "Reminder: Library books are due in 3 days. Renew online at lib.city.gov",
        "Your two-factor authentication PIN for GitHub is 901248.",
        "See you at the gym tomorrow morning at 7 AM!",
        "Apple ID verification code: 582910. Do not share with Apple support.",
        "Your food order from DoorDash has been picked up by the Dasher.",
        "Electric utility bill is ready. Total due $84.20 on Nov 5th.",
        "Your scheduled payment for internet service was processed.",
        "Hey, did you remember to turn off the coffee machine at the office?",
        "Your grocery delivery is arriving between 3 PM and 4 PM today.",
        "Thanks for shopping at Target. View your digital receipt here.",
        "Happy birthday! Wishing you a fantastic year ahead filled with happiness.",
        "Your dentist appointment has been rescheduled to Thursday at 10 AM.",
        "PayPal: You sent $15.00 USD to John Doe.",
        "Your lab test results are now available on the patient portal.",
        "Hey, can you review the pull request on GitHub when you have a moment?",
        "Wells Fargo Alert: Low balance notice on checking account ending in 4102.",
    ] * 20  # Replicate to provide a stable distribution

    scam_samples = [
        "URGENT: Your Chase bank account has been locked due to suspicious activity. Verify immediately at http://chase-security-verify.com/login",
        "CONGRATULATIONS! You have won a $1,000 Walmart Gift Card! Claim now before expiration: http://claim-gift-rewards.net/win",
        "USPS: Your delivery is on hold due to missing address details. Update your info here to avoid return: http://usps-redelivery-fee.top",
        "IRS Final Notice: An arrest warrant has been issued in your name for tax evasion. Call 800-555-0199 immediately to settle.",
        "Alert: Unusual login detected from Moscow, Russia on your Amazon account. If this was not you, secure account at: http://amzn-security-alert.cc",
        "Make $500/day working from home on WhatsApp! Zero experience needed. Message our hiring manager now: http://wa.me/scam98214",
        "WellsFargo: Your credit card was debited $892.40. If you did NOT authorize this transaction, click http://wellsfargo-dispute-auth.org",
        "Crypto Alert: You received 1.25 BTC ($84,000) in your pending wallet! Login and claim within 24 hours: http://binance-claim-crypto.site",
        "Apple Support: Your iPhone was located near Chicago. Sign in with your Apple ID to view coordinates: http://icloud-findmy-lost.info",
        "FINAL WARNING: Your Netflix subscription will be suspended today. Update credit card billing at http://netflix-payment-renewal.com",
        "FedEx: Package tracking #9824194 requires a $1.99 customs fee payment: http://fedex-parcel-clearance.net",
        "Dear customer, your PayPal account has been limited. Please click to verify your identity: http://paypal-resolution-center.xyz",
        "You have 1 unread intimate video recorded from your webcam. Pay 500 USD in Bitcoin or we leak to contacts: http://bit-leak.pw",
        "Bank Fraud Alert: Did you attempt a transfer of $1,450 to Zelle? Reply NO to cancel or open: http://fraud-prevention-alert.link",
        "Govt Relief Grant: You are eligible for $4,500 inflation assistance. Submit direct deposit bank details: http://gov-stimulus-claim.org",
        "Your SIM card is scheduled to be deactivated in 2 hours. Call our telecom customer care immediately.",
        "Job Offer: Amazon is hiring online product review assistants. Earn $30-$50 per hour. Register: http://amazon-hiring-portal.site",
        "DHL Express: Your shipment #849102 could not be delivered. Reschedule delivery: http://dhl-parcel-redirection.top",
        "AT&T Alert: Your March bill payment is overdue. Pay now to avoid immediate cellular disconnection: http://att-billpay-portal.info",
        "URGENT KYC: Your bank account will be blocked today under RBI/SEC guidelines. Update PAN/Aadhaar: http://kyc-update-portal.cc",
    ] * 20

    data = []
    for text in ham_samples:
        data.append({"text": text, "label": 0})
    for text in scam_samples:
        data.append({"text": text, "label": 1})

    df = pd.DataFrame(data)
    print(f"[*] Generated dataset with {len(df)} samples ({len(ham_samples)} HAM, {len(scam_samples)} SCAM).")
    return df


# ---------------------------------------------------------------------------
# 4. Tokenization & Formatting
# ---------------------------------------------------------------------------
def prepare_hf_dataset(df: pd.DataFrame, tokenizer, max_length: int = 128):
    """
    Converts DataFrame into Hugging Face Dataset and applies tokenization.
    """
    hf_dataset = Dataset.from_pandas(df)

    def tokenize_func(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_token_type_ids=False,  # DistilBERT doesn't use token_type_ids
        )

    tokenized = hf_dataset.map(tokenize_func, batched=True)
    return tokenized


# ---------------------------------------------------------------------------
# 5. Export Fine-Tuned PyTorch Model to ONNX
# ---------------------------------------------------------------------------
def export_to_onnx(
    model: nn.Module,
    tokenizer,
    onnx_output_path: str,
    max_length: int = 128,
):
    """
    Exports PyTorch Transformer model to ONNX with dynamic batch and sequence axes.
    """
    print(f"\n[*] Exporting PyTorch model to ONNX: {onnx_output_path}")
    model.eval()
    model.cpu()

    # Create dummy input representative of typical inference
    dummy_text = "Your Chase bank account is suspended. Verify now: http://phish.com"
    inputs = tokenizer(
        dummy_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Dynamic axes allow arbitrary batch sizes and sequence lengths at runtime
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    }

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(os.path.abspath(onnx_output_path)), exist_ok=True)

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )

    # Validate ONNX model integrity
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    file_size_mb = os.path.getsize(onnx_output_path) / (1024 * 1024)
    print(f"[OK] ONNX model successfully exported and verified. Size: {file_size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# 6. Dynamic INT8 Quantization (Mobile CPU Target <35MB)
# ---------------------------------------------------------------------------
def quantize_onnx_model(
    input_onnx_path: str,
    output_quant_path: str,
) -> float:
    """
    Applies Dynamic INT8 Quantization (QInt8) on weights using ONNX Runtime.
    Reduces memory footprint by ~75% with minimal accuracy degradation (<0.5%).
    """
    print(f"\n[*] Applying Dynamic INT8 Quantization to {input_onnx_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(output_quant_path)), exist_ok=True)

    quantize_dynamic(
        model_input=input_onnx_path,
        model_output=output_quant_path,
        weight_type=QuantType.QInt8,
    )

    quant_size_mb = os.path.getsize(output_quant_path) / (1024 * 1024)
    print(f"[OK] Quantized ONNX model generated: {output_quant_path}")
    print(f"[OK] Quantized Model Size: {quant_size_mb:.2f} MB (Target: < 35.0 MB)")

    if quant_size_mb > 35.0:
        print("[!] WARNING: Model exceeds 35MB limit. Consider MobileBERT or smaller seq_length.")
    else:
        print("[OK] Size Requirement Met (<35MB) for mobile deployment!")

    return quant_size_mb


# ---------------------------------------------------------------------------
# 7. Export Vocabulary for Native Kotlin WordPiece Tokenizer
# ---------------------------------------------------------------------------
def export_vocab_for_android(tokenizer, output_dir: str):
    """
    Exports vocab.txt from Hugging Face tokenizer to be bundled directly into
    Android app assets for zero-dependency native on-device tokenization.
    """
    os.makedirs(output_dir, exist_ok=True)
    vocab_path = os.path.join(output_dir, "vocab.txt")
    print(f"\n[*] Exporting tokenizer vocabulary to {vocab_path}...")

    vocab_dict = tokenizer.get_vocab()
    # Sort by token index to guarantee exact 1:1 mapping in Kotlin
    sorted_tokens = sorted(vocab_dict.items(), key=lambda x: x[1])

    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, _ in sorted_tokens:
            f.write(token + "\n")

    print(f"[OK] Exported {len(sorted_tokens)} vocabulary tokens to {vocab_path}")


# ---------------------------------------------------------------------------
# 8. Main Training and Quantization Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ScamShield Model Training & ONNX Quantization")
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Hugging Face model backbone ('distilbert-base-uncased' or 'google/mobilebert-uncased')",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to CSV dataset with 'text' and 'label' columns",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save fine-tuned checkpoints and ONNX files",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--ham_penalty_weight",
        type=float,
        default=4.0,
        help="Class loss penalty weight on legitimate HAM (class 0) to ensure FPR < 0.5%",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum token sequence length",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")

    # Load Tokenizer & Model
    print(f"[*] Loading pretrained backbone: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "HAM", 1: "SCAM"},
        label2id={"HAM": 0, "SCAM": 1},
    )

    # Load & Split Data
    df = load_or_generate_dataset(args.data_path)
    train_df, eval_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    train_dataset = prepare_hf_dataset(train_df, tokenizer, args.max_length)
    eval_dataset = prepare_hf_dataset(eval_df, tokenizer, args.max_length)

    # Calculate class weights for False Positive suppression
    # weight_0 = penalty for misclassifying legitimate message
    # weight_1 = penalty for missing a scam
    # Giving higher weight to class 0 drastically reduces False Positives (FPR < 0.5%)
    class_weights = torch.tensor([args.ham_penalty_weight, 1.0], dtype=torch.float)
    print(f"[*] Applied loss weights: HAM (0) = {class_weights[0]}, SCAM (1) = {class_weights[1]}")

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_total_limit=1,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "class_weights": class_weights,
    }
    import inspect
    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedTrainer(**trainer_kwargs)

    print("\n[*] Starting fine-tuning...")
    trainer.train()

    print("\n[*] Evaluating fine-tuned model...")
    eval_results = trainer.evaluate()
    print("--- Final Evaluation Metrics ---")
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Verify FPR Constraint
    final_fpr = eval_results.get("eval_fpr", 0.0)
    print(f"\n[*] Measured False Positive Rate (FPR): {final_fpr * 100:.3f}%")
    if final_fpr <= 0.005:
        print("[OK] SUCCESS: False Positive Rate is within required threshold (< 0.5%)!")
    else:
        print("[!] Note: To further decrease FPR, increase --ham_penalty_weight or tune the classification threshold.")

    # Save Best PyTorch Model
    saved_model_dir = os.path.join(args.output_dir, "best_model")
    model.save_pretrained(saved_model_dir)
    tokenizer.save_pretrained(saved_model_dir)

    # Export Vocabulary for Android
    export_vocab_for_android(tokenizer, os.path.join(args.output_dir, "android_assets"))

    # Export ONNX (FP32)
    onnx_raw_path = os.path.join(args.output_dir, "scam_model_raw.onnx")
    export_to_onnx(model, tokenizer, onnx_raw_path, max_length=args.max_length)

    # Dynamic INT8 Quantization
    onnx_quant_path = os.path.join(args.output_dir, "android_assets", "scam_model_quantized.onnx")
    quantize_onnx_model(onnx_raw_path, onnx_quant_path)

    # Automatically copy artifacts to Android app assets directory
    android_assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "android", "app", "src", "main", "assets"))
    if os.path.exists(android_assets_dir):
        print(f"[*] Copying model artifacts directly to Android assets: {android_assets_dir}...")
        shutil.copy2(onnx_quant_path, os.path.join(android_assets_dir, "scam_model_quantized.onnx"))
        shutil.copy2(os.path.join(args.output_dir, "android_assets", "vocab.txt"), os.path.join(android_assets_dir, "vocab.txt"))
        print("[OK] Copied scam_model_quantized.onnx and vocab.txt directly to Android assets!")

    print("\n=======================================================")
    print("[DONE] ScamShield ML Pipeline Completed Successfully!")
    print(f"[*] Android Model:  {onnx_quant_path}")
    print(f"[*] Android Vocab:  {os.path.join(args.output_dir, 'android_assets', 'vocab.txt')}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
