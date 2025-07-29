#!/bin/bash

# Define paths
BASE_MODEL_PATH="/home/ray/mike/energy_guard/gpt2/gpt2_aegis_model/checkpoint-best-loss/gpt2_aegis_model_filtered.bin"
OUTPUT_DIR="/home/ray/mike/energy_guard/gpt2/continual_learning_results"
ATTACK_TYPE="AIM_attack"
ATTACK_DIR="/home/ray/mike/energy_guard/gpt2/aegis_dataset/attacked_bins_batches/$ATTACK_TYPE"
TRAIN_FILE="$ATTACK_DIR/train_${ATTACK_TYPE}.csv"
JAILBREAK_FILE="$ATTACK_DIR/test_${ATTACK_TYPE}.csv"  # Renamed for clarity but file name remains the same
VALIDATION_FILE="/home/ray/mike/energy_guard/gpt2/aegis_dataset/validation_only_data.csv"
MODEL_NAME_OR_PATH="gpt2"
TOKENIZER_NAME="gpt2"

# Create output directory and results directory if they don't exist
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/results

# Check if the train and jailbreak files exist
if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Train file not found at $TRAIN_FILE"
    echo "Please run combine_attack_files.sh first"
    exit 1
fi

if [ ! -f "$JAILBREAK_FILE" ]; then
    echo "Error: Jailbreak file not found at $JAILBREAK_FILE"
    echo "Please run combine_attack_files.sh first"
    exit 1
fi

# Count the number of examples in train file
NUM_EXAMPLES=$(wc -l < $TRAIN_FILE)
NUM_EXAMPLES=$((NUM_EXAMPLES - 1))  # Subtract 1 for header row

echo "Total examples in train file: $NUM_EXAMPLES"
echo "Using jailbreak evaluation file: $JAILBREAK_FILE"
echo "Using validation file: $VALIDATION_FILE"

# Limit the number of examples to process (for faster testing)
MAX_EXAMPLES=50
if [ $NUM_EXAMPLES -gt $MAX_EXAMPLES ]; then
    NUM_EXAMPLES=$MAX_EXAMPLES
    echo "Limiting to $MAX_EXAMPLES examples for processing"
fi

# Initial model is the base model
CURRENT_MODEL_PATH=$BASE_MODEL_PATH

# Run iteration 0: Evaluate base model without any training
echo "Running iteration 0 (base model evaluation without training)"

# Define the model name for iteration 0
ITER_MODEL_NAME="gpt2_aegis_model_${ATTACK_TYPE}_iter_0.bin"

# Run the continual learning evaluation for iteration 0 (no training, just evaluation)
python main_energy_attack.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --tokenizer_name $TOKENIZER_NAME \
    --do_continual_learning \
    --model_name $ITER_MODEL_NAME \
    --base_model_path $CURRENT_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --batch1_file $TRAIN_FILE \
    --test_data_file $JAILBREAK_FILE \
    --validation_file $VALIDATION_FILE \
    --continual_learning_examples 0 \
    --iteration 0 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 5e-5 \
    --epochs 3 \
    --block_size 512 \
    --_lambda 1.0

echo "Completed iteration 0 (base model evaluation). Results saved."
echo "---------------------------------------------------------"

# Run continual learning with increasing number of examples starting from 1
for ((i=1; i<=$NUM_EXAMPLES; i+=1)); do
    echo "Running iteration with $i examples"
    
    # Define the model name for this iteration
    ITER_MODEL_NAME="gpt2_aegis_model_${ATTACK_TYPE}_iter_$i.bin"
    
    # Run the continual learning evaluation
    python main_energy_attack.py \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --tokenizer_name $TOKENIZER_NAME \
        --do_continual_learning \
        --model_name $ITER_MODEL_NAME \
        --base_model_path $CURRENT_MODEL_PATH \
        --output_dir $OUTPUT_DIR \
        --batch1_file $TRAIN_FILE \
        --test_data_file $JAILBREAK_FILE \
        --validation_file $VALIDATION_FILE \
        --continual_learning_examples $i \
        --iteration $i \
        --train_batch_size 4 \
        --eval_batch_size 4 \
        --learning_rate 5e-5 \
        --epochs 3 \
        --block_size 512 \
        --_lambda 1.0
    
    # Update the current model path for the next iteration
    CURRENT_MODEL_PATH="$OUTPUT_DIR/continual-learning-checkpoint/$ITER_MODEL_NAME"
    
    echo "Completed iteration with $i examples. Model saved to $CURRENT_MODEL_PATH"
    echo "---------------------------------------------------------"
done

echo "Continual learning evaluation completed!"
echo "Results saved in $OUTPUT_DIR/results/" 