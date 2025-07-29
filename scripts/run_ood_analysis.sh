#!/bin/bash

# Directory containing the result files
RESULTS_DIR="../results"
OUTPUT_BASE_DIR="../results/ood_analysis_results"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_BASE_DIR"

# Define the metrics to evaluate
METRICS=("energy_scores" "mahalanobis_distances" "likelihood_ratios" "uncertainties")

# Log file for overall script execution
SCRIPT_LOG="$OUTPUT_BASE_DIR/run_ood_analysis_$(date +%Y%m%d_%H%M%S).log"
echo "Starting OOD detection analysis at $(date)" | tee -a "$SCRIPT_LOG"
echo "Results will be saved to $OUTPUT_BASE_DIR" | tee -a "$SCRIPT_LOG"

# Find all training files with 'inference_training' in the name
TRAINING_FILES=$(find "$RESULTS_DIR" -name "*inference_training*.csv")
echo "Found $(echo "$TRAINING_FILES" | wc -l) training files to process" | tee -a "$SCRIPT_LOG"

# Loop through each training file
for TRAIN_FILE in $TRAINING_FILES; do
    # Extract the parameter pattern from the filename
    # Example: gpt2_aegis_model_lr5e-5_lambda3_norm2.0_inference_training.csv
    BASE_NAME=$(basename "$TRAIN_FILE")
    PARAM_PATTERN=$(echo "$BASE_NAME" | sed 's/inference_training.*\.csv//')
    
    # Construct the corresponding test file name with 'combined_test' instead of 'inference_training'
    TEST_FILE=$(find "$RESULTS_DIR" -name "${PARAM_PATTERN}combined_test*.csv" | head -1)
    
    if [ -z "$TEST_FILE" ]; then
        echo "Warning: No matching test file found for $TRAIN_FILE" | tee -a "$SCRIPT_LOG"
        continue
    fi
    
    echo "========================================================" | tee -a "$SCRIPT_LOG"
    echo "Processing model: $PARAM_PATTERN" | tee -a "$SCRIPT_LOG"
    echo "Training file: $TRAIN_FILE" | tee -a "$SCRIPT_LOG"
    echo "Test file: $TEST_FILE" | tee -a "$SCRIPT_LOG"
    
    # Create a specific output directory for this model
    MODEL_OUTPUT_DIR="$OUTPUT_BASE_DIR/${PARAM_PATTERN}results"
    mkdir -p "$MODEL_OUTPUT_DIR"
    
    # Create a log file for this model
    MODEL_LOG="$MODEL_OUTPUT_DIR/analysis_log.txt"
    echo "Analysis for model $PARAM_PATTERN started at $(date)" > "$MODEL_LOG"
    echo "Training file: $TRAIN_FILE" >> "$MODEL_LOG"
    echo "Test file: $TEST_FILE" >> "$MODEL_LOG"
    
    # Run analysis for each metric individually
    for METRIC in "${METRICS[@]}"; do
        echo "" | tee -a "$SCRIPT_LOG" "$MODEL_LOG"
        echo "Analyzing with metric: $METRIC" | tee -a "$SCRIPT_LOG" "$MODEL_LOG"
        
        # Create metric-specific output directory
        METRIC_OUTPUT_DIR="$MODEL_OUTPUT_DIR/$METRIC"
        mkdir -p "$METRIC_OUTPUT_DIR"
        
        # Create a log file for this specific metric analysis
        METRIC_LOG="$METRIC_OUTPUT_DIR/analysis_log.txt"
        
        # Execute the analysis script and capture output to log file
        echo "Running analysis for $METRIC at $(date)" | tee -a "$MODEL_LOG" "$METRIC_LOG"
        {
            python "$RESULTS_DIR/ood_detection_analysis.py" \
                --train_file "$TRAIN_FILE" \
                --test_file "$TEST_FILE" \
                --metrics "$METRIC" \
                --quantile 95 \
                --output_dir "$METRIC_OUTPUT_DIR" 2>&1
        } | tee -a "$MODEL_LOG" "$METRIC_LOG"
        
        echo "Completed analysis for $METRIC at $(date)" | tee -a "$SCRIPT_LOG" "$MODEL_LOG" "$METRIC_LOG"
    done
    
    echo "Analysis complete for $PARAM_PATTERN at $(date)" | tee -a "$SCRIPT_LOG" "$MODEL_LOG"
    echo "Results saved to $MODEL_OUTPUT_DIR" | tee -a "$SCRIPT_LOG" "$MODEL_LOG"
    echo "========================================================" | tee -a "$SCRIPT_LOG"
done

# Generate a summary report of all results
echo "Generating summary report..." | tee -a "$SCRIPT_LOG"
SUMMARY_FILE="$OUTPUT_BASE_DIR/ood_detection_summary.csv"
echo "model,metric,jailbreak_detection_rate,false_positive_rate" > "$SUMMARY_FILE"

# Find all result CSV files and extract key metrics
find "$OUTPUT_BASE_DIR" -name "ood_detection_results_*.csv" | while read -r RESULT_FILE; do
    MODEL_PATTERN=$(echo "$RESULT_FILE" | grep -o "gpt2_aegis_model_[^/]*" | head -1)
    METRIC=$(basename "$RESULT_FILE" | grep -o "ood_detection_results_[^_]*" | sed 's/ood_detection_results_//')
    
    # Extract detection rates
    JAILBREAK_RATE=$(grep "jailbreak_detection_rate" "$RESULT_FILE" | cut -d',' -f2)
    FP_RATE=$(grep "false_positive_rate" "$RESULT_FILE" | cut -d',' -f2)
    
    # Add to summary
    echo "$MODEL_PATTERN,$METRIC,$JAILBREAK_RATE,$FP_RATE" >> "$SUMMARY_FILE"
done

echo "Complete! Summary report saved to $SUMMARY_FILE" | tee -a "$SCRIPT_LOG"
echo "OOD detection analysis finished at $(date)" | tee -a "$SCRIPT_LOG" 