#!/bin/bash

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

# Base directory for attack data
BASE_DIR="/home/ray/mike/energy_guard/gpt2/aegis_dataset/attacked_bins_batches"
# Create logs directory
mkdir -p logs

# Loop through each attack type
for attack in "${attack_types[@]}"; do
    echo "Processing $attack..."
    
    # Create the result folder if it doesn't exist
    RESULT_DIR="$BASE_DIR/$attack"
    mkdir -p "$RESULT_DIR"
    
    # Set paths for test data and results
    TEST_DATA="aegis_dataset/attacked_bins_batches/${attack}/processed_combined_${attack}.csv"
    RESULT_FILE="$RESULT_DIR/test_results.csv"
    
    # Check if test data file exists
    if [ ! -f "$TEST_DATA" ]; then
        echo "Warning: Test data file $TEST_DATA not found, skipping $attack"
        continue
    fi
    
    echo "Running test for $attack with data file $TEST_DATA"
    
    # Run the test for this attack type
    python main_energy_attack.py \
        --do_test \
        --train_data_file aegis_dataset/labeled_first_category/aegis_train_100_or_less.csv \
        --eval_data_file aegis_dataset/labeled_first_category/aegis_validation.csv \
        --test_data_file "$TEST_DATA" \
        --output_dir gpt2_aegis_model \
        --model_type gpt2 \
        --model_name_or_path gpt2 \
        --tokenizer_name gpt2 \
        --model_name gpt2_aegis_model_filtered.bin \
        --save_test_result_filepath "$RESULT_FILE" \
        --block_size 512 \
        --train_batch_size 8 \
        --eval_batch_size 8 \
        --learning_rate 1e-4 \
        --_lambda 0.5 \
        --max_grad_norm 2.0 \
        --epochs 10 \
        --seed 42 \
        --evaluate_during_training 2>&1 | tee "logs/gpt2_aegis_testing_${attack}.log"
    
    echo "Completed processing $attack"
    echo "----------------------------------------"
done



    