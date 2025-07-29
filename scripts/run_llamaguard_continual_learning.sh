#!/bin/bash

# Base directory containing all attack folders
BASE_DIR="/home/ray/mike/energy_guard/gpt2/aegis_dataset/attacked_bins_batches"
OUTPUT_DIR="/home/ray/mike/energy_guard/gpt2/llamaguard_continual_learning_results"
SCRIPT_PATH="/home/ray/mike/energy_guard/gpt2/llamaguard_continual_learning.py"

# Create output directory and results directory if they don't exist
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/results
mkdir -p $OUTPUT_DIR/results_csv

# List of attack types
attack_types=(
    "AIM_attack"
    "base64_attack"
    "caesar_attack"
    "code_chameleon_attack"
    "combination_attack" 
    "DAN_attack"
    "deepInception_attack"
    "self_cipher_attack"
    "SmartGPT_attack"
    "zulu_attack"
)

# Limit the number of examples to process (for faster testing)
MAX_EXAMPLES=50

# Use the 1B model
LLAMAGUARD_MODEL="meta-llama/Llama-Guard-3-1B"
VALIDATION_FILE="/home/ray/mike/energy_guard/gpt2/aegis_dataset/validation_only_data.csv"

for ATTACK_TYPE in "${attack_types[@]}"; do
    ATTACK_DIR="$BASE_DIR/$ATTACK_TYPE"
    FILE_PREFIX=$ATTACK_TYPE
    TRAIN_FILE="$ATTACK_DIR/train_${FILE_PREFIX}.csv"
    TEST_FILE="$ATTACK_DIR/test_${FILE_PREFIX}.csv"

    # Check if the train and test files exist
    if [ ! -f "$TRAIN_FILE" ]; then
        echo "Error: Train file not found at $TRAIN_FILE"
        continue
    fi
    if [ ! -f "$TEST_FILE" ]; then
        echo "Error: Test file not found at $TEST_FILE"
        continue
    fi

    # Count the number of examples in train file
    NUM_EXAMPLES=$(wc -l < $TRAIN_FILE)
    NUM_EXAMPLES=$((NUM_EXAMPLES - 1))  # Subtract 1 for header row
    if [ $NUM_EXAMPLES -gt $MAX_EXAMPLES ]; then
        NUM_EXAMPLES=$MAX_EXAMPLES
    fi

    CURRENT_ITER_MODEL_PATH=""

    for ((i=1; i<=$NUM_EXAMPLES; i+=1)); do
        echo "Running iteration with example #$i for $ATTACK_TYPE (LlamaGuard 1B)"
        ITER_OUTPUT_DIR="$OUTPUT_DIR/$ATTACK_TYPE/iter_$i"
        mkdir -p "$ITER_OUTPUT_DIR/results_csv"

        # Extract the i-th example from the train file
        python -c "
import pandas as pd
import os
train_file = '$TRAIN_FILE'
out_file = '$ITER_OUTPUT_DIR/temp_batch_$i.csv'
df = pd.read_csv(train_file)
if len(df) < $i:
    print(f'Error: Not enough rows in the CSV file. Need at least {$i} data rows.')
    exit(1)
example = df.iloc[$i-1:$i]
example.to_csv(out_file, index=False)
if not os.path.exists(out_file) or os.path.getsize(out_file) == 0:
    print('Error: Failed to create valid batch file')
    exit(1)
df_check = pd.read_csv(out_file)
if len(df_check) != 1:
    print(f'Warning: Expected 1 example row but got {len(df_check)}')
    if len(df_check) == 0:
        print('Error: No data rows in batch file')
        exit(1)
else:
    print(f'Successfully created batch file with 1 example (row {$i})')
"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create batch file for iteration $i"
            continue
        fi

        # Build the command with optional base model path
        CMD_ARGS=(
            --train_data_file "$ITER_OUTPUT_DIR/temp_batch_$i.csv"
            --test_data_file "$TEST_FILE"
            --eval_data_file "$VALIDATION_FILE"
            --output_dir "$ITER_OUTPUT_DIR"
            --model_name_or_path "$LLAMAGUARD_MODEL"
            --epochs 1
            --learning_rate 1e-4
            --batch_size 1
            --iteration $i
        )

        # 🔄 CONTINUAL LEARNING: Always try to load from previous iteration
        PREV_ITER_MODEL_PATH="$OUTPUT_DIR/$ATTACK_TYPE/iter_$((i-1))/model_iter_$((i-1))"
        if [ $i -gt 1 ] && [ -d "$PREV_ITER_MODEL_PATH" ]; then
            CMD_ARGS+=(--base_model_path "$PREV_ITER_MODEL_PATH")
            echo "🔄 Continuing from iteration $((i-1)): $PREV_ITER_MODEL_PATH"
        else
            echo "🆕 Starting with base model (iteration $i)"
        fi

        # Run the continual learning script for this example
        python "$SCRIPT_PATH" "${CMD_ARGS[@]}"
        
        # Check if training was successful by verifying model was saved
        CURRENT_ITER_MODEL_PATH="$ITER_OUTPUT_DIR/model_iter_$i"
        if [ ! -d "$CURRENT_ITER_MODEL_PATH" ]; then
            echo "⚠️  Warning: Model not saved for iteration $i, continual learning chain may be broken"
        else
            echo "✅ Model saved successfully for iteration $i"
        fi

        # Clean up temp file
        rm -f "$ITER_OUTPUT_DIR/temp_batch_$i.csv"
    done

    # Optionally, aggregate results for this attack
    echo "Aggregating results for $ATTACK_TYPE"
    python -c "
import os
import json
import pandas as pd
results_dir = '$OUTPUT_DIR/$ATTACK_TYPE'
all_data = []
for subdir, dirs, files in os.walk(results_dir):
    for file in files:
        if file.endswith('.json'):
            with open(os.path.join(subdir, file), 'r') as f:
                data = json.load(f)
                all_data.append(data)
if all_data:
    df = pd.DataFrame(all_data)
    summary_file = os.path.join(results_dir, 'results_summary.csv')
    df.to_csv(summary_file, index=False)
    print(f'Summary saved to {summary_file}')
else:
    print('No results found to summarize')
"
done

echo "All attack types processed!"
echo "All results saved in $OUTPUT_DIR/" 