import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import argparse
import logging
import numpy as np
import random
import os
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set environment variables to avoid accelerate issues
os.environ["ACCELERATE_USE_FSDP"] = "0"
os.environ["ACCELERATE_USE_DEEPSPEED"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def transform_dataset_prompt(model_name_or_path, user_prompt):
    if "Llama-Guard-3-1B" in model_name_or_path:
        # LlamaGuard-3-1B expects nested content format
        chat = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": user_prompt
                    },
                ],
            }
        ]
        return chat
    elif "Llama-Guard-3-8B" in model_name_or_path:
        # LlamaGuard-3-8B uses simple format
        chat = [
            {"role": "user", "content": user_prompt}
        ]
        return chat
    elif "gpt2" in model_name_or_path.lower():
        # For GPT-2, format as a safety classification task
        safety_prompt = f"Classify the following prompt as safe or unsafe:\n\nPrompt: {user_prompt}\n\nClassification:"
        chat = [
            {"role": "user", "content": safety_prompt},
        ]
        return chat
    else:
        # For original LlamaGuard-7b and other models
        chat = [
            {"role": "user", "content": user_prompt},
        ]
        return chat

def tokenize_with_template(tokenizer, user_inputs, agent_labels, max_length, model_name_or_path):
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    
    # Reserve space for the response (label) tokens
    max_input_length = max_length - 20  # Reserve 20 tokens for response
    
    for user_input, agent_label in zip(user_inputs, agent_labels):
        chat = transform_dataset_prompt(model_name_or_path, user_input)
        
        if "gpt2" in model_name_or_path.lower():
            # For GPT-2, directly tokenize the formatted prompt
            prompt_text = chat[0]["content"]
            full_text = prompt_text + " " + agent_label
            
            # Tokenize prompt and full text separately
            prompt_tokens = tokenizer(
                prompt_text, 
                return_tensors="pt", 
                truncation=True,
                add_special_tokens=True
            )["input_ids"].squeeze(0)
            
            full_encoding = tokenizer(
                full_text, 
                return_tensors="pt", 
                max_length=max_length, 
                padding="max_length", 
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True
            )
            
            input_ids = full_encoding["input_ids"].squeeze(0)
            attention_mask = full_encoding["attention_mask"].squeeze(0)
            
            # Create labels - mask the prompt part, keep the response part
            labels = torch.full_like(input_ids, -100)
            prompt_len = len(prompt_tokens)
            if prompt_len < len(input_ids):
                # Only set labels for the response part
                response_start = min(prompt_len, len(input_ids) - 1)
                labels[response_start:] = input_ids[response_start:]
                # Mask padding tokens in labels
                labels[attention_mask == 0] = -100
            
        else:
            # For LlamaGuard models - use chat template
            # First, apply chat template to get the input (with reserved space)
            input_ids = tokenizer.apply_chat_template(
                chat, 
                return_tensors="pt", 
                add_generation_prompt=True,  # This adds the assistant prompt
                truncation=True,
                max_length=max_input_length  # Use reduced length to leave space for labels
            ).squeeze(0)
            
            # Create proper label format for LlamaGuard
            if agent_label.lower() == "unsafe":
                # For unsafe, use a simple category
                label_text = "unsafe\nS1"
            else:
                label_text = "safe"
            
            # Tokenize the label
            label_tokens = tokenizer(
                label_text, 
                return_tensors="pt", 
                add_special_tokens=False,
                truncation=True
            )["input_ids"].squeeze(0)
            
            # Combine input and label
            combined_ids = torch.cat([input_ids, label_tokens])
            
            # Pad to max_length (should not need truncation now)
            if len(combined_ids) > max_length:
                # If still too long, truncate the input part but keep all label tokens
                input_truncate_len = max_length - len(label_tokens)
                if input_truncate_len > 0:
                    input_ids_truncated = input_ids[:input_truncate_len]
                    combined_ids = torch.cat([input_ids_truncated, label_tokens])
                else:
                    # Extreme case: just use the label tokens
                    combined_ids = label_tokens[:max_length]
            
            # Pad if needed
            if len(combined_ids) < max_length:
                pad_length = max_length - len(combined_ids)
                pad_tokens = torch.full((pad_length,), tokenizer.pad_token_id, dtype=combined_ids.dtype)
                combined_ids = torch.cat([combined_ids, pad_tokens])
            
            # Create attention mask
            attention_mask = (combined_ids != tokenizer.pad_token_id).long()
            
            # Create labels - mask input part, keep label part
            labels = torch.full_like(combined_ids, -100)
            input_len = len(input_ids) if len(input_ids) <= len(combined_ids) else len(combined_ids) - len(label_tokens)
            
            # Set labels for the response part only
            if input_len < len(combined_ids):
                label_start = input_len
                label_end = min(input_len + len(label_tokens), len(combined_ids))
                if label_start < label_end:
                    labels[label_start:label_end] = combined_ids[label_start:label_end]
            
            # Ensure we have at least some non-masked labels
            valid_labels_count = (labels != -100).sum()
            if valid_labels_count == 0:
                # Emergency fallback: unmask the last few non-padding tokens
                valid_tokens = (combined_ids != tokenizer.pad_token_id).nonzero().flatten()
                if len(valid_tokens) > 0:
                    # Unmask the last few tokens (likely the label)
                    last_tokens = valid_tokens[-min(len(label_tokens), len(valid_tokens)):]
                    labels[last_tokens] = combined_ids[last_tokens]
            
            input_ids = combined_ids
            
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)
    
    encodings = {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(attention_mask_list)
    }
    return encodings

def evaluate_on_file(model, tokenizer, data_file, args, split_name="eval", iteration=1):
    df = pd.read_csv(data_file)
    texts = df["prompt"].tolist() if "prompt" in df else df.iloc[:,0].tolist()
    original_labels = df["prompt_label"].tolist() if "prompt_label" in df else ["unsafe"] * len(texts)
    
    model.eval()
    binary_preds = []
    binary_trues = []
    all_outputs = []
    
    # Convert original labels to binary
    for label in original_labels:
        binary_trues.append(1 if label.lower() == "unsafe" else 0)
    
    print(f"\n🔍 Evaluating on {split_name} set ({len(texts)} examples)...")
    
    with torch.no_grad():
        for i, (text, label) in enumerate(tqdm(zip(texts, original_labels), desc=f"Evaluating {split_name}", total=len(texts))):
            # Create chat format for evaluation
            chat = transform_dataset_prompt(args.model_name_or_path, text)
            
            # Apply chat template for generation
            if "gpt2" in args.model_name_or_path.lower():
                prompt_text = chat[0]["content"]
                input_ids = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_train_input_length
                ).input_ids.to(model.device)
            else:
                # For LlamaGuard models
                input_ids = tokenizer.apply_chat_template(
                    chat,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    truncation=True,
                    max_length=args.max_train_input_length
                ).to(model.device)
            
            # Generate with better parameters
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=1,
                    do_sample=False,  # Use greedy decoding
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    attention_mask=(input_ids != tokenizer.pad_token_id).long(),
                    use_cache=True
                )
            
            # Decode only the generated part
            prompt_len = input_ids.shape[1]
            if len(outputs[0]) > prompt_len:
                generated_tokens = outputs[0][prompt_len:]
                output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            else:
                output_text = ""
            
            pred = 1 if "unsafe" in output_text.lower() else 0

            binary_preds.append(pred)
            all_outputs.append(output_text)
    
    # Calculate metrics
    acc = accuracy_score(binary_trues, binary_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(binary_trues, binary_preds, average='binary', zero_division=0)
    dsr = sum((p == 1 and t == 1) for p, t in zip(binary_preds, binary_trues)) / (sum(t == 1 for t in binary_trues) or 1)
    
    # Count statistics
    total_unsafe = sum(t == 1 for t in binary_trues)
    total_safe = sum(t == 0 for t in binary_trues)
    correct_unsafe = sum((p == 1 and t == 1) for p, t in zip(binary_preds, binary_trues))
    correct_safe = sum((p == 0 and t == 0) for p, t in zip(binary_preds, binary_trues))
    
    # Display detailed stats
    print(f"\n📊 {split_name.upper()} SET RESULTS (Iteration {iteration}):")
    print("=" * 60)
    print(f"📈 Overall Metrics:")
    print(f"   • Binary Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"   • Precision:       {precision:.4f}")
    print(f"   • Recall:          {recall:.4f}")
    print(f"   • F1-Score:        {f1:.4f}")
    print(f"   • DSR:             {dsr:.4f} ({dsr*100:.2f}%)")
    print(f"\n🎯 Classification Breakdown:")
    print(f"   • Total Examples:     {len(binary_trues)}")
    print(f"   • Unsafe Examples:    {total_unsafe}")
    print(f"   • Safe Examples:      {total_safe}")
    print(f"   • Correct Unsafe:     {correct_unsafe}/{total_unsafe} ({correct_unsafe/total_unsafe*100 if total_unsafe > 0 else 0:.1f}%)")
    print(f"   • Correct Safe:       {correct_safe}/{total_safe} ({correct_safe/total_safe*100 if total_safe > 0 else 0:.1f}%)")
    print("=" * 60)
    
    # Save CSV
    df["binary_preds"] = binary_preds
    df["model_outputs"] = all_outputs
    csv_path = os.path.join(args.output_dir, "results_csv", f"{split_name}_iter_{iteration}_predictions.csv")
    df.to_csv(csv_path, index=False)
    
    return {
        f"{split_name}_binary_acc": float(acc),
        f"{split_name}_binary_precision": float(precision),
        f"{split_name}_binary_recall": float(recall),
        f"{split_name}_binary_f1": float(f1),
        f"{split_name}_dsr": float(dsr)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_file", type=str, required=True)
    parser.add_argument("--test_data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-Guard-3-1B")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)  # Reduced from 1e-3
    parser.add_argument("--epochs", type=int, default=3)  # Increased from 1
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_train_input_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--eval_data_file", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--base_model_path", type=str, default=None, help="Path to previously trained model")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "results_csv"), exist_ok=True)

    # Load data
    train_df = pd.read_csv(args.train_data_file)
    test_df = pd.read_csv(args.test_data_file)
    train_texts = train_df["prompt"].tolist() if "prompt" in train_df else train_df.iloc[:,0].tolist()
    train_labels = ["unsafe"] * len(train_texts)

    test_texts = test_df["prompt"].tolist() if "prompt" in test_df else test_df.iloc[:,0].tolist()
    test_labels = test_df["prompt_label"].tolist() if "prompt_label" in test_df else ["unsafe"] * len(test_texts)

    print(f"\n🚀 LLAMAGUARD CONTINUAL LEARNING - ITERATION {args.iteration}")
    print("=" * 80)
    print(f"📂 Training Data: {args.train_data_file} ({len(train_texts)} examples)")
    print(f"📂 Test Data: {args.test_data_file} ({len(test_texts)} examples)")
    if args.eval_data_file:
        print(f"📂 Validation Data: {args.eval_data_file}")
    print(f"🤖 Model: {args.model_name_or_path}")
    print(f"⚙️  Training Config: {args.epochs} epochs, LR={args.learning_rate}, Batch={args.batch_size}")
    if args.base_model_path:
        print(f"🔄 Continuing from: {args.base_model_path}")
    print("=" * 80)

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model - either base model or continue from previous iteration
    if args.base_model_path and os.path.exists(args.base_model_path):
        print(f"🔄 Loading model from previous iteration: {args.base_model_path}")
        try:
            # Load the saved LoRA model
            from peft import PeftModel
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            model = PeftModel.from_pretrained(base_model, args.base_model_path)
            
            # CRITICAL: Ensure only LoRA parameters are trainable
            # First, freeze all base model parameters
            for param in model.base_model.parameters():
                param.requires_grad = False
            
            # Then enable training only for LoRA parameters
            model.train()
            for name, param in model.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            print("✅ Successfully loaded previous model and enabled LoRA training mode")
        except Exception as e:
            print(f"⚠️  Failed to load previous model: {e}")
            print("🔄 Loading base model instead...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            # Apply LoRA config
            lora_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none"
            )
            model = get_peft_model(model, lora_config)
    else:
        print("🆕 Loading base model...")
        # Load base model with conservative settings
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_cache=False,
                trust_remote_code=True,
                _fast_init=False
            )
        except Exception as e:
            print(f"Failed to load model with standard method: {e}")
            print("Trying alternative loading method...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        
        # Apply LoRA config
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        model = get_peft_model(model, lora_config)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"🔥 Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("💻 Using CPU")
    
    print(f"\n🔧 LoRA Configuration:")
    model.print_trainable_parameters()

    # Prepare data
    print(f"\n📝 Preparing training data...")
    train_encodings = tokenize_with_template(tokenizer, train_texts, train_labels, args.max_train_input_length, args.model_name_or_path)
    train_dataset = torch.utils.data.TensorDataset(
        train_encodings["input_ids"], 
        train_encodings["labels"],
        train_encodings["attention_mask"]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Training loop (continual learning handled by bash script between iterations)
    print(f"\n🎯 Starting Training (Continual Learning)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    model.train()
    
    total_loss = 0
    valid_batches = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_valid_batches = 0
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(bar):
            optimizer.zero_grad()
            input_ids, labels, attention_mask = batch
            input_ids = input_ids.to(model.device)
            labels = labels.to(model.device)
            attention_mask = attention_mask.to(model.device)
            
            # Check if we have valid labels (not all -100)
            valid_labels = (labels != -100).any()
            if not valid_labels:
                print(f"⚠️  Skipping batch {batch_idx} - no valid labels")
                continue
            
            outputs = model(
                input_ids=input_ids, 
                labels=labels,
                attention_mask=attention_mask
            )
            loss = outputs.loss
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  NaN/Inf loss detected in batch {batch_idx}, skipping...")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            
            epoch_loss += loss.item()
            total_loss += loss.item()
            epoch_valid_batches += 1
            valid_batches += 1
            
            avg_loss = epoch_loss / max(epoch_valid_batches, 1)
            bar.set_description(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")
    
    avg_loss = total_loss / max(valid_batches, 1)
    print(f"\n✅ Training Complete! Average Loss: {avg_loss:.4f} (Valid batches: {valid_batches})")

    # Save the trained model for next iteration
    model_save_path = os.path.join(args.output_dir, f"model_iter_{args.iteration}")
    try:
        model.save_pretrained(model_save_path)
        print(f"💾 Model saved to: {model_save_path}")
    except Exception as e:
        print(f"⚠️  Failed to save model: {e}")

    # Evaluation on test set
    print(f"\n🧪 Starting Evaluation...")
    test_results = evaluate_on_file(model, tokenizer, args.test_data_file, args, split_name="test", iteration=args.iteration)
    
    # Evaluation on validation set if provided
    eval_results = {}
    if args.eval_data_file:
        eval_results = evaluate_on_file(model, tokenizer, args.eval_data_file, args, split_name="eval", iteration=args.iteration)
    
    # Save JSON
    results = {
        "attack_type": "llamaguard_continual",
        "num_examples": len(train_dataset),
        "iteration": args.iteration,
        "avg_training_loss": float(avg_loss),
        "valid_training_batches": valid_batches,
        **test_results,
        **eval_results
    }
    json_path = os.path.join(args.output_dir, f"llamaguard_iter_{args.iteration}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Final summary
    print(f"\n🎉 ITERATION {args.iteration} COMPLETE!")
    print("=" * 80)
    print(f"📊 FINAL SUMMARY:")
    print(f"   • Training Loss:     {avg_loss:.4f}")
    print(f"   • Valid Batches:     {valid_batches}")
    print(f"   • Test DSR:          {test_results.get('test_dsr', 0):.4f} ({test_results.get('test_dsr', 0)*100:.2f}%)")
    print(f"   • Test F1:           {test_results.get('test_binary_f1', 0):.4f}")
    print(f"   • Test Accuracy:     {test_results.get('test_binary_acc', 0):.4f} ({test_results.get('test_binary_acc', 0)*100:.2f}%)")
    if eval_results:
        print(f"   • Validation DSR:    {eval_results.get('eval_dsr', 0):.4f} ({eval_results.get('eval_dsr', 0)*100:.2f}%)")
        print(f"   • Validation F1:     {eval_results.get('eval_binary_f1', 0):.4f}")
        print(f"   • Validation Acc:    {eval_results.get('eval_binary_acc', 0):.4f} ({eval_results.get('eval_binary_acc', 0)*100:.2f}%)")
    print(f"💾 Results saved to: {json_path}")
    print(f"💾 Model saved to: {model_save_path}")
    print("=" * 80)
    
    logger.info(f"Saved results to {json_path}")

if __name__ == "__main__":
    main() 