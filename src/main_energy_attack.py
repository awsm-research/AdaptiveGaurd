from __future__ import absolute_import, division, print_function
import argparse
import glob
import logging
import os
import pickle
import random
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import get_linear_schedule_with_warmup, GPT2Model, AutoTokenizer, get_constant_schedule
from torch.optim import AdamW
from tqdm import tqdm
from .model import Model
import pandas as pd
# metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset
from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_normal
from torch.nn.functional import softmax
import json
import time

cpu_cont = 16
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_ids,
                 label):
        self.input_ids = input_ids
        self.label=label
        

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_type="train"):
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "eval":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        elif file_type == "retrain":
            file_path = args.retrain_data_file
            
        self.examples = []
        
        if file_type == "ood_train":
            df = pd.read_csv("../data/train_JailBreakV-28k.csv")
            prompts = df["jailbreak_query"].tolist()
        elif file_type == "ood_val":
            df = pd.read_csv("../data/val_JailBreakV-28k.csv")
            prompts = df["jailbreak_query"].tolist()
        ### iterate through all attack types ###
        elif file_type == "test":
            # df_1 = pd.read_csv("../split_attack_prompts/AIM_data_1_percent/all_data.csv")
            # df_2 = pd.read_csv("../split_attack_prompts/base64_data_1_percent/all_data.csv")
            # df_3 = pd.read_csv("../split_attack_prompts/caesar_cipher_data_1_percent/all_data.csv")
            # df_4 = pd.read_csv("../split_attack_prompts/CC_data_1_percent/all_data.csv")
            # df_5 = pd.read_csv("../split_attack_prompts/combination_data_1_percent/all_data.csv")
            # df_6 = pd.read_csv("../split_attack_prompts/DAN_data_1_percent/all_data.csv")
            # df_7 = pd.read_csv("../split_attack_prompts/deepInception_data_1_percent/all_data.csv")
            # df_8 = pd.read_csv("../split_attack_prompts/dual_use_data_1_percent/all_data.csv")
            # df_9 = pd.read_csv("../split_attack_prompts/self_cipher_data_1_percent/all_data.csv")
            # df_10 = pd.read_csv("../split_attack_prompts/zulu_data_1_percent/all_data.csv")
            # df = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10]).reset_index(drop=True)
            df = pd.read_csv(file_path)
            if "transformed_prompt" in df.columns:
                prompts = df["transformed_prompt"].tolist()
            else:
                prompts = df["prompt"].tolist()
            # prompts = df["transformed_prompt"].tolist()#[:200]
            labels = df["label_index"].tolist()#[:200]
            df.to_csv(f"./total_test_cases.csv")

        
        else:
            df = pd.read_csv(file_path)
            prompts = df["prompt"].tolist()#[:200]
            labels = df["label_index"].tolist()#[:200]

            # df.to_csv(f"./total_train_cases.csv")
        
        for i in tqdm(range(len(prompts))):
            if file_type == "ood_train" or file_type == "ood_val":
                self.examples.append(convert_examples_to_features(prompts[i], 0, tokenizer, args))
            else:
                self.examples.append(convert_examples_to_features(prompts[i], labels[i], tokenizer, args))

        # if file_type == "train":
        #     for example in self.examples[:3]:
        #         logger.info("*** Example ***")
        #         logger.info("label: {}".format(example.label))
        #         logger.info("type label: {}".format(example.type_label))
        #         logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)

def convert_examples_to_features(prompt, label, tokenizer, args):    
    input_ids = tokenizer(
                            str(prompt),
                            padding='max_length',
                            truncation=True,
                            max_length=args.block_size,
                         ).input_ids
    return InputFeatures(input_ids, label)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, ood_train_dataset, ood_val_dataset, model, tokenizer, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)
    
    # random sampling with replacement to align num of samples in OOD with ID
    ood_sampler = RandomSampler(ood_train_dataset, replacement=True, num_samples=len(train_dataset))
    ood_dataloader = DataLoader(ood_train_dataset, sampler=ood_sampler, batch_size=args.train_batch_size, num_workers=0)

    args.max_steps = args.epochs * len(train_dataloader)
    # evaluate the model per epoch
    args.save_steps = len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    #scheduler = get_constant_schedule(optimizer)
    
    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step=0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_loss = 1e6
    avg_energy = 0
    model.zero_grad()
    
    for idx in range(args.epochs): 
        tr_num = 0
        train_loss = 0
        total_energy = 0
        
        bar = tqdm(enumerate(zip(train_dataloader, ood_dataloader)), 
               total=min(len(train_dataloader), len(ood_dataloader)),)
        
        for step, (batch, ood_batch) in bar:
            (input_ids, labels) = [x.to(args.device) for x in batch]
            (ood_input_ids, _) = [x.to(args.device) for x in ood_batch]  # OOD batch has no labels
 
            model.train()

            # Get model outputs
            bin_loss, energy_score = model(input_ids=input_ids, labels=labels)
            ood_energy_score = model(input_ids=ood_input_ids, return_energy_only=True)[-1]  # Get energy for OOD data

            # Compute squared hinge loss for OOD regularization
            min_energy, max_energy = -30, 0
            loss_in = torch.mean(torch.square(torch.relu(energy_score - min_energy)))
            loss_out = torch.mean(torch.square(torch.relu(max_energy - ood_energy_score)))
            
            # Total OOD loss
            ood_loss = loss_in + loss_out
            
            # Add energy suppression loss with a very small lambda
            target_energy = -30.0  # Set a low target energy
            energy_loss = torch.mean(torch.square(energy_score - target_energy))
            small_lambda = 0.01
            loss = bin_loss + small_lambda * energy_loss

            if args.n_gpu > 1:
                loss = loss.mean()
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            total_energy += torch.mean(energy_score).item()
            tr_num += 1
            train_loss += loss.item()
            
            if avg_loss == 0:
                avg_loss = tr_loss
                
            avg_loss = round(train_loss/tr_num,5)
            avg_energy = round(total_energy / tr_num, 5)
            bar.set_description("epoch {} loss {} energy {}".format(idx, avg_loss, avg_energy))
              
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer, eval_dataset, ood_val_dataset, eval_when_training=True)    
                    
                    # Save model checkpoint
                    if results < best_loss:
                        best_loss = results
                        logger.info("  "+"*"*20)  
                        logger.info("  Best Loss:%s",round(best_loss,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-loss'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name)) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        
def evaluate(args, model, tokenizer, eval_dataset, ood_val_dataset, eval_when_training=False):
    # Build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # random sampling with replacement to align num of samples in OOD with ID
    ood_sampler = SequentialSampler(ood_val_dataset)
    ood_dataloader = DataLoader(ood_val_dataset, sampler=ood_sampler, batch_size=args.train_batch_size, num_workers=0)

    # Multi-GPU evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss = 0
    num_batches = 0
    for step, (batch, ood_batch) in enumerate(zip(eval_dataloader, ood_dataloader)):
        (input_ids, labels) = [x.to(args.device) for x in batch]
        (ood_input_ids, _) = [x.to(args.device) for x in ood_batch]  # OOD batch has no labels
        with torch.no_grad():
            bin_loss, energy_score = model(input_ids=input_ids, labels=labels)
            ood_energy_score = model(input_ids=ood_input_ids, return_energy_only=True)[-1]  # Get energy for OOD data
            
            # Compute squared hinge loss for OOD regularization
            min_energy, max_energy = -23, -5
            loss_in = torch.mean(torch.square(torch.relu(energy_score - min_energy)))
            loss_out = torch.mean(torch.square(torch.relu(max_energy - ood_energy_score)))

            # Total OOD loss
            ood_loss = loss_in + loss_out
            eval_loss += ood_loss.item()
            
        num_batches += 1
    
    eval_loss = eval_loss / num_batches
    
    # Logging results
    logger.info("***** Eval results *****")
    logger.info(f"Eval OOD Loss: {eval_loss}")

    return eval_loss

def test(args, model, tokenizer, eval_dataset, eval_when_training=False):
    # Build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # Multi-GPU evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    binary_preds = []  # For binary
    binary_trues = []  # For binary

    for batch in eval_dataloader:
        (input_ids, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            prob, _ = model(input_ids=input_ids)

            batch_binary_preds = list(torch.argmax(prob, dim=1).cpu().numpy())            
            # Predictions and truths for binary classification
            binary_preds += batch_binary_preds
            binary_trues += list(labels.cpu().numpy().flatten())            

    # Binary metrics
    binary_acc = accuracy_score(binary_trues, binary_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(binary_trues, binary_preds, average='binary')

    result = {
        "eval_binary_acc": float(binary_acc),
        "eval_binary_precision": float(precision),
        "eval_binary_recall": float(recall),
        "eval_binary_f1": float(f1),
    }

    # Logging results
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    # write to csv file
    df = pd.read_csv(args.test_data_file)
    df["binary_preds"] = binary_preds
    model_name = args.model_name.replace(".bin", "")
    df.to_csv(f"./{model_name}_predictions.csv")
    
    return result

def energy_loss(energy_scores, target_energy=0.2):
    """
    Penalizes high energy scores to force OOD samples to have lower energy.
    Uses Mean Squared Error (MSE) between current energy scores and target value.
    """
    return torch.mean((energy_scores - target_energy) ** 2)  # MSE Loss to bring energy scores down

def energy_based_test_with_rolling_updates(
    args, model, tokenizer, eval_dataset, energy_threshold, eval_when_training=False
):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running Energy-Based evaluation with rolling updates *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    y_preds, y_trues, binary_preds, binary_trues, accepted_indices = [], [], [], [], []
    all_energy_scores = []  # To collect energy scores for computing mean and median
    
    OOD_samples_limit = 5
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    OOD_remaining = 0
    
    # important note: this implementation assumes that batch size = 1
    for batch in bar:
        OOD_detected = False
        
        input_ids, labels = [x.to(args.device) for x in batch]

        model.eval()
        with torch.no_grad():
            prob, energy_scores = model(input_ids=input_ids)
        model.train()
                
        # Detach energy_scores to avoid gradients
        energy_scores = energy_scores.cpu().tolist()
        all_energy_scores += energy_scores
                
        # Collect OOD samples for a single step update
        ood_input_ids, ood_labels = [], []
        for idx, (e, inp, lbl) in enumerate(zip(energy_scores, input_ids, labels)):
            if e > energy_threshold:  # OOD detected
                ood_input_ids.append(inp)
                ood_labels.append(lbl)
                OOD_remaining += 1
                OOD_detected = True

            # Process predictions for metrics
            binary_preds.append(torch.argmax(prob[idx]).item())
            binary_trues.append(lbl.item())
        
        """
        if OOD_detected:
            logger.info("OOD DETECTED!")
            
            if OOD_samples_limit > 0:
                ood_input_ids = torch.stack(ood_input_ids).to(args.device)
                ood_labels = torch.stack(ood_labels).to(args.device)
                ood_type_labels = torch.stack(ood_type_labels).to(args.device)

                num_iterations = 1  # Number of times to train on the same OOD batch
                for _ in range(num_iterations):
                    # Forward pass
                    bin_loss, type_loss, energy_score = model(input_ids=ood_input_ids, labels=ood_labels, type_labels=ood_type_labels)
                    
                    target_energy = -10  # Ideal target value for ID samples
                    energy_loss = torch.mean((energy_score - target_energy) ** 2)
                    
                    # Update total loss function
                    loss = bin_loss + type_loss + energy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            OOD_samples_limit -= 1  # Reduce limit after training
            OOD_remaining -= 1
        """
    # Metrics considering all samples
    multi_class_acc = accuracy_score(y_trues, y_preds)
    binary_acc = accuracy_score(binary_trues, binary_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(binary_trues, binary_preds, average="binary")

    result = {
        "eval_multi_class_acc": float(multi_class_acc),
        "eval_binary_acc": float(binary_acc),
        "eval_binary_precision": float(precision),
        "eval_binary_recall": float(recall),
        "eval_binary_f1": float(f1),
    }

    logger.info("***** Energy-Based Eval results with rolling updates *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
    
    median = np.quantile(all_energy_scores, q=0.5)
    
    q95 = np.quantile(all_energy_scores, q=0.95)
    
    q99 = np.quantile(all_energy_scores, q=0.99)
    
    avg = sum(all_energy_scores)/len(all_energy_scores)
    
    logger.info(f"median {median}")
    logger.info(f"q95 {q95}")
    logger.info(f"q99 {q99}")
    logger.info(f"avg {avg}")
    logger.info(f"min {min(all_energy_scores)}")
    logger.info(f"max {max(all_energy_scores)}")
    logger.info("OOD samples still remaining %d", OOD_remaining)
    
    with open("OOD_all_energy_scores.pkl", "wb+") as f:
        pickle.dump(all_energy_scores, f)
    
    """
    df = pd.read_csv(args.test_data_file)
    df["binary_preds"] = binary_preds
    df["multi_class_preds"] = y_preds
    df['energy_scores'] = all_energy_scores
    model_name = args.model_name.replace(".bin", "")
    df.to_csv(f"./{model_name}_energy_predictions_attack.csv")
    """
    
    return all_energy_scores


def compute_likelihood_ratio(input_ids, model, tokenizer):
    """Computes likelihood ratio for OOD detection."""
    with torch.no_grad():
        logits = model(input_ids)[0]
    probs = softmax(logits, dim=-1).cpu().numpy()
    log_likelihood = np.log(probs + 1e-10).sum(axis=-1)  # Avoid log(0)
    return log_likelihood

def ensemble_uncertainty(input_ids, model, num_passes=5):
    """Computes uncertainty based on ensemble variance using dropout."""
    model.train()
    predictions = []
    for _ in range(num_passes):
        with torch.no_grad():
            logits = model(input_ids)[0]
            probs = softmax(logits, dim=-1).cpu().numpy()
        predictions.append(probs)
    model.eval()
    predictions = np.array(predictions)
    return predictions.var(axis=0).mean()
def compute_mahalanobis_stats_binary(model, dataset, device, batch_size):
    """
    Computes:
      - mu_0: Mean penultimate-layer feature for 'safe' class
      - mu_1: Mean penultimate-layer feature for 'unsafe' class
      - cov_inv: Inverse of the shared covariance over both classes
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    features_by_class = {0: [], 1: []}

    with torch.no_grad():
        for input_ids, labels in tqdm(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Forward pass to extract hidden states
            prob, energy_score, all_layers = model(
                input_ids, 
                return_hidden_state=True
            )
            penultimate_layer = all_layers[-2]  # shape = [batch_size, seq_len, hidden_dim]

            feats = penultimate_layer.mean(dim=1).cpu().numpy()

            for f, label in zip(feats, labels):
                # label.item() should be 0 or 1
                features_by_class[label.item()].append(f)

    # Stack features for each class
    arr_0 = np.stack(features_by_class[0])  # safe
    arr_1 = np.stack(features_by_class[1])  # unsafe

    # Compute means
    mu_0 = arr_0.mean(axis=0)  # shape [hidden_dim]
    mu_1 = arr_1.mean(axis=0)  # shape [hidden_dim]

    # Compute shared covariance across both classes
    all_feats = np.vstack([arr_0, arr_1])
    overall_mean = all_feats.mean(axis=0)
    centered = all_feats - overall_mean
    cov = np.cov(centered, rowvar=False) + np.eye(centered.shape[1]) * 1e-6
    cov_inv = np.linalg.inv(cov)

    return mu_0, mu_1, cov_inv

def improved_ood_detection(args, model, tokenizer, eval_dataset, train_dataset, energy_threshold, eval_when_training=False):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)
    model.eval()
    all_energy_scores, all_mahalanobis_distances, all_likelihood_ratios, all_uncertainties = [], [], [], []
    binary_preds = []
    # 1) Get mean & cov_inv from in-distribution training
    mu_0, mu_1, cov_inv = compute_mahalanobis_stats_binary(
        model,
        train_dataset,
        args.device,
        batch_size=args.eval_batch_size
    )
    for batch in eval_dataloader:
        input_ids, _ = [x.to(args.device) for x in batch]
        batch_size = input_ids.size(0)  # Get the actual batch size
        with torch.no_grad():
            prob, energy_score, all_layers = model(
                input_ids,
                return_hidden_state=True
            )
            # batch predictions
            batch_binary_preds = torch.argmax(prob, dim=1).cpu().numpy()
            binary_preds.extend(batch_binary_preds.tolist())
            # Get energy scores for each sample in the batch
            energy_scores = energy_score.cpu().numpy()
            # Get likelihood ratios for each sample in the batch
            likelihood_ratios = compute_likelihood_ratio(input_ids, model, tokenizer)
            # Compute uncertainty for each sample individually to ensure correct dimensions
            batch_uncertainties = []
            for i in range(batch_size):
                single_input = input_ids[i:i+1]  # Get a single sample
                uncertainty = ensemble_uncertainty(single_input, model)
                batch_uncertainties.append(uncertainty)
            # Extract Mahalanobis features
            penultimate_layer = all_layers[-2]  # shape = [batch_size, seq_len, hidden_dim]
            feats = penultimate_layer.mean(dim=1).cpu().numpy()
            mahalanobis_distances = []
            for f in feats:
                dist_0 = mahalanobis(f, mu_0, cov_inv)
                dist_1 = mahalanobis(f, mu_1, cov_inv)
                mahalanobis_distances.append(min(dist_0, dist_1))
        # Add all metrics ensuring they're properly aligned with individual samples
        all_energy_scores.extend(energy_scores)
        all_mahalanobis_distances.extend(mahalanobis_distances)
        all_likelihood_ratios.extend(likelihood_ratios)
        all_uncertainties.extend(batch_uncertainties)
    result = {
        "energy_scores": all_energy_scores,
        "mahalanobis_distances": all_mahalanobis_distances,
        "likelihood_ratios": all_likelihood_ratios,
        "uncertainties": all_uncertainties
    }
    logger.info("***** OOD detection metrics *****")
    # for key in sorted(result.keys()):
    #     logger.info("  %s = %s", key, str(round(result[key], 4)))
    median = np.quantile(all_energy_scores, q=0.5)
    q95 = np.quantile(all_energy_scores, q=0.95)
    q99 = np.quantile(all_energy_scores, q=0.99)
    avg = sum(all_energy_scores)/len(all_energy_scores)
    logger.info(f"median {median}")
    logger.info(f"q95 {q95}")
    logger.info(f"q99 {q99}")
    logger.info(f"avg {avg}")
    logger.info(f"min {min(all_energy_scores)}")
    logger.info(f"max {max(all_energy_scores)}")
    logger.info("***** Mahalanobis distance metrics *****")
    # for key in sorted(result.keys()):
    #     logger.info("  %s = %s", key, str(round(result[key], 4)))
    median = np.quantile(all_mahalanobis_distances, q=0.5)
    q95 = np.quantile(all_mahalanobis_distances, q=0.95)
    q99 = np.quantile(all_mahalanobis_distances, q=0.99)
    avg = sum(all_mahalanobis_distances)/len(all_mahalanobis_distances)
    logger.info(f"median {median}")
    logger.info(f"q95 {q95}")
    logger.info(f"q99 {q99}")
    logger.info(f"avg {avg}")
    logger.info(f"min {min(all_mahalanobis_distances)}")
    logger.info(f"max {max(all_mahalanobis_distances)}")
    logger.info("***** Likelihood ratio metrics *****")
    # for key in sorted(result.keys()):
    #     logger.info("  %s = %s", key, str(round(result[key], 4)))
    median = np.quantile(all_likelihood_ratios, q=0.5)
    q95 = np.quantile(all_likelihood_ratios, q=0.95)
    q99 = np.quantile(all_likelihood_ratios, q=0.99)
    avg = sum(all_likelihood_ratios)/len(all_likelihood_ratios)
    logger.info(f"median {median}")
    logger.info(f"q95 {q95}")
    logger.info(f"q99 {q99}")
    logger.info(f"avg {avg}")
    logger.info(f"min {min(all_likelihood_ratios)}")
    logger.info(f"max {max(all_likelihood_ratios)}")
    logger.info("***** Uncertainty metrics *****")
    median = np.quantile(all_uncertainties, q=0.5)
    q95 = np.quantile(all_uncertainties, q=0.95)
    q99 = np.quantile(all_uncertainties, q=0.99)
    avg = sum(all_uncertainties)/len(all_uncertainties)
    logger.info(f"median {median}")
    logger.info(f"q95 {q95}")
    logger.info(f"q99 {q99}")
    logger.info(f"avg {avg}")
    logger.info(f"min {min(all_uncertainties)}")
    logger.info(f"max {max(all_uncertainties)}")
    # Verify all metrics have the same length
    assert len(binary_preds) == len(all_energy_scores) == len(all_mahalanobis_distances) == len(all_likelihood_ratios) == len(all_uncertainties)
    logger.info(f"Total evaluated samples: {len(binary_preds)}")
    with open("OOD_all_train_mahalanobis_distances.pkl", "wb+") as f:
        pickle.dump(all_mahalanobis_distances, f)
    with open("OOD_all_train_likelihood_ratios.pkl", "wb+") as f:
        pickle.dump(all_likelihood_ratios, f)
    with open("OOD_all_train_uncertainties.pkl", "wb+") as f:
        pickle.dump(all_uncertainties, f)
    df = pd.read_csv(args.test_data_file)
    # Ensure we have the correct number of predictions
    if len(df) != len(binary_preds):
        logger.warning(f"Warning: DataFrame length ({len(df)}) doesn't match number of predictions ({len(binary_preds)}). This could cause misalignment.")
        exit()
    df["binary_preds"] = binary_preds
    df['energy_scores'] = all_energy_scores
    df['mahalanobis_distances'] = all_mahalanobis_distances
    df['likelihood_ratios'] = all_likelihood_ratios
    df['uncertainties'] = all_uncertainties
    model_name = args.model_name.replace(".bin", "")
    if args.save_test_result_filepath:
        df.to_csv(args.save_test_result_filepath)
    else:
        df.to_csv(f"./{model_name}_test_results.csv")
    return result


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_retrain", action='store_true',default=False,
                        help="Whether to run retraining.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_continual_learning", action='store_true',
                        help="Whether to run continual learning evaluation.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_token_level_eval", default=False, action='store_true',
                        help="Whether to do local explanation.") 
    parser.add_argument("--reasoning_method", default="attention", type=str,
                        help="Should be one of 'attention', 'shap', 'lime', 'lig'")
    parser.add_argument("--save_test_result_filepath", default= None, type=str,
                        help="Path to save the test results.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")
    parser.add_argument('--safe_prompt_index', type=int, default=0,
                        help="")
    parser.add_argument("--_lambda", default=1.0, type=float,
                        help="")
    
    # Continual learning specific parameters
    parser.add_argument("--batch1_file", default=None, type=str,
                        help="Path to training file for continual learning")
    parser.add_argument("--validation_file", default=None, type=str,
                        help="Path to validation file for continual learning")
    parser.add_argument("--continual_learning_examples", default=1, type=int,
                        help="Number of examples to use from training file for continual learning")
    parser.add_argument("--base_model_path", default=None, type=str,
                        help="Path to the base model for continual learning")
    parser.add_argument("--iteration", default=1, type=int,
                        help="Current iteration of continual learning")
       
    args = parser.parse_args()

    
    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)
    
    # Set seed
    set_seed(args)
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2Model.from_pretrained(args.model_name_or_path)
    
    model = Model(model, tokenizer, args, num_labels=2) # 10 unsafe + 1 safe

    logger.info("Training/evaluation parameters %s", args)
    
    # Continual Learning Evaluation
    if args.do_continual_learning:
        # Create results directory if it doesn't exist
        results_dir = os.path.join(args.output_dir, 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Load the base model
        if args.base_model_path:
            model.load_state_dict(torch.load(args.base_model_path, map_location=args.device), strict=False)
            model.to(args.device)
            logger.info(f"Loaded base model from {args.base_model_path}")
        
        # Load batch 1 for training
        batch1_df = pd.read_csv(args.batch1_file)
        
        # Take only the specified number of examples for this iteration
        current_examples = min(args.continual_learning_examples, len(batch1_df))
        
        # Only train if we have examples to train on
        if current_examples > 0:
            batch1_subset_df = batch1_df.iloc[:current_examples]
            
            # Create a temporary CSV file with the subset
            temp_train_file = f"temp_train_{args.iteration}.csv"
            batch1_subset_df.to_csv(temp_train_file, index=False)
            
            # Set the train data file to the temporary file
            args.train_data_file = temp_train_file
            
            # Create training dataset
            train_dataset = TextDataset(tokenizer, args, file_type='train')
            
            # Fine-tune the model on the training data
            model = continual_learning_train(args, model, tokenizer, train_dataset)
            
            # Clean up temporary file
            os.remove(temp_train_file)
        else:
            logger.info(f"Skipping training for iteration {args.iteration} (0 examples)")
        
        # Load jailbreak dataset for evaluation
        jailbreak_dataset = None
        if args.test_data_file:
            args.test_data_file = args.test_data_file
            jailbreak_dataset = TextDataset(tokenizer, args, file_type='test')
        
        # Evaluate DSR on jailbreak dataset
        jailbreak_results = {}
        if jailbreak_dataset:
            jailbreak_results = evaluate_dsr(args, model, tokenizer, jailbreak_dataset)
        
        # Evaluate on validation dataset
        validation_results = {}
        if args.validation_file:
            args.test_data_file = args.validation_file
            validation_dataset = TextDataset(tokenizer, args, file_type='test')
            
            # Run standard test evaluation
            validation_results = test(args, model, tokenizer, validation_dataset)
            
            # Rename keys to use 'validation' prefix
            validation_results = {f"validation_{k.replace('eval_', '')}": v for k, v in validation_results.items()}
        
        # Combine results
        all_results = {
            "attack_type": os.path.basename(os.path.dirname(args.batch1_file)),
            "num_examples": args.continual_learning_examples,
            "iteration": args.iteration,
            **jailbreak_results,
            **validation_results
        }
        
        # Save results to file in the results directory
        attack_type = os.path.basename(os.path.dirname(args.batch1_file))
        results_file = os.path.join(results_dir, f"{attack_type}_iter_{args.iteration}.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Saved results to {results_file}")
        
        return
    
    # Training
    if args.do_retrain:
        train_dataset = TextDataset(tokenizer, args, file_type='retrain')
        ood_train_dataset =  TextDataset(tokenizer, args, file_type='ood_train')
        
        eval_dataset = TextDataset(tokenizer, args, file_type='eval')
        ood_val_dataset =  TextDataset(tokenizer, args, file_type='ood_val')
        
        args.retrain_mode = True
        train(args, train_dataset, ood_train_dataset, ood_val_dataset, model, tokenizer, eval_dataset)

    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, file_type='train')
        ood_train_dataset =  TextDataset(tokenizer, args, file_type='ood_train')
        
        eval_dataset = TextDataset(tokenizer, args, file_type='eval')
        ood_val_dataset =  TextDataset(tokenizer, args, file_type='ood_val')
        
        train(args, train_dataset, ood_train_dataset, ood_val_dataset, model, tokenizer, eval_dataset)

    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-loss/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, map_location=args.device),strict=False)
        model.to(args.device)
        
        # for energy threshold
        train_dataset = TextDataset(tokenizer, args, file_type='train')
        
        test_dataset = TextDataset(tokenizer, args, file_type='test')
        
        results = improved_ood_detection(args, model, tokenizer, test_dataset,train_dataset, energy_threshold=-23)

def continual_learning_train(args, model, tokenizer, train_dataset):
    """Train the model on a single batch for continual learning"""
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)
    
    args.max_steps = args.epochs * len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    
    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running continual learning training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    
    global_step=0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    avg_energy = 0
    model.zero_grad()
    
    for idx in range(args.epochs): 
        tr_num = 0
        train_loss = 0
        total_energy = 0
        
        for step, batch in enumerate(train_dataloader):
            (input_ids, labels) = [x.to(args.device) for x in batch]
            model.train()

            # Get model outputs
            bin_loss, energy_score = model(input_ids=input_ids, labels=labels)
            
            # Add energy suppression loss with a very small lambda
            target_energy = -30.0  # Set a low target energy
            energy_loss = torch.mean(torch.square(energy_score - target_energy))
            small_lambda = 0.003
            loss = bin_loss + small_lambda * energy_loss

            if args.n_gpu > 1:
                loss = loss.mean()
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            total_energy += torch.mean(energy_score).item()
            tr_num += 1
            train_loss += loss.item()
            
            if avg_loss == 0:
                avg_loss = tr_loss
                
            avg_loss = round(train_loss/tr_num,5)
            avg_energy = round(total_energy / tr_num, 5)
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                
            # Log energy scores periodically
            if step % 5 == 0:
                logger.info(f"Epoch {idx}, Step {step}: Average Energy = {avg_energy}, Loss = {avg_loss}")
    
    # Save the model after training
    output_dir = os.path.join(args.output_dir, 'continual-learning-checkpoint')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model,'module') else model
    output_dir = os.path.join(output_dir, '{}'.format(args.model_name))
    torch.save(model_to_save.state_dict(), output_dir)
    logger.info("Saving continual learning model checkpoint to %s", output_dir)
    logger.info(f"Final average energy score: {avg_energy}")
    
    return model

def evaluate_dsr(args, model, tokenizer, test_dataset):
    """Evaluate Defence Success Rate on jailbreak dataset"""
    logger.info("***** Evaluating DSR on jailbreak dataset *****")
    
    # Build dataloader
    eval_sampler = SequentialSampler(test_dataset)
    eval_dataloader = DataLoader(test_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # Multi-GPU evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.eval()
    binary_preds = []
    binary_trues = []
    all_energy_scores = []

    for batch in eval_dataloader:
        (input_ids, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            prob, energy_score = model(input_ids=input_ids)
            
            # Get predictions
            batch_binary_preds = list(torch.argmax(prob, dim=1).cpu().numpy())
            binary_preds += batch_binary_preds
            binary_trues += list(labels.cpu().numpy().flatten())
            
            # Collect energy scores
            energy_scores = energy_score.cpu().tolist()
            all_energy_scores += energy_scores

    # Calculate metrics
    binary_acc = accuracy_score(binary_trues, binary_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(binary_trues, binary_preds, average='binary')
    
    # Calculate DSR
    # Assuming unsafe is labeled as 1, we want to correctly identify unsafe examples
    unsafe_indices = [i for i, label in enumerate(binary_trues) if label == 1]
    correct_unsafe_predictions = sum(1 for i in unsafe_indices if binary_preds[i] == 1)
    dsr = correct_unsafe_predictions / len(unsafe_indices) if unsafe_indices else 0
    
    result = {
        "jailbreak_binary_acc": float(binary_acc),
        "jailbreak_precision": float(precision),
        "jailbreak_recall": float(recall),
        "jailbreak_f1": float(f1),
        "jailbreak_dsr": float(dsr),
        "jailbreak_energy_mean": float(sum(all_energy_scores) / len(all_energy_scores)),
        "jailbreak_energy_min": float(min(all_energy_scores)),
        "jailbreak_energy_max": float(max(all_energy_scores)),
    }
    
    # Logging results
    logger.info("***** Jailbreak Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
    
    return result

if __name__ == "__main__":
    main()