#!/bin/bash

# Base directory containing all attack folders
BASE_DIR="../data/aegis_dataset/attacked_bins_batches"

# Process each attack folder
for attack_dir in "$BASE_DIR"/*_attack; do
    if [ -d "$attack_dir" ]; then
        attack_name=$(basename "$attack_dir")
        echo "Processing $attack_name..."
        
        # Path for combined file
        COMBINED_FILE="$attack_dir/combined_${attack_name}.csv"
        
        # Check if combined file already exists
        if [ -f "$COMBINED_FILE" ]; then
            echo "Combined file already exists at $COMBINED_FILE. Skipping."
            continue
        fi
        
        # Get header from first batch file and write to combined file
        first_batch=$(ls "$attack_dir"/batch_*.csv | head -n 1)
        if [ -z "$first_batch" ]; then
            echo "No batch files found in $attack_dir. Skipping."
            continue
        fi
        
        echo "Creating combined file at $COMBINED_FILE"
        head -n 1 "$first_batch" > "$COMBINED_FILE"
        
        # Append data from all batch files (skipping headers)
        for batch_file in "$attack_dir"/batch_*.csv; do
            echo "  Adding data from $(basename "$batch_file")"
            tail -n +2 "$batch_file" >> "$COMBINED_FILE"
        done
        
        # Count total lines
        total_lines=$(wc -l < "$COMBINED_FILE")
        data_lines=$((total_lines - 1))  # Subtract header line
        echo "  Combined file has $data_lines data rows"
        
        # Create train and test splits (last 50 samples for testing)
        if [ $data_lines -gt 50 ]; then
            train_lines=$((data_lines - 50))
            
            TRAIN_FILE="$attack_dir/train_${attack_name}.csv"
            TEST_FILE="$attack_dir/test_${attack_name}.csv"
            
            echo "  Creating train file with $train_lines samples at $TRAIN_FILE"
            head -n 1 "$COMBINED_FILE" > "$TRAIN_FILE"
            tail -n +2 "$COMBINED_FILE" | head -n $train_lines >> "$TRAIN_FILE"
            
            echo "  Creating test file with 50 samples at $TEST_FILE"
            head -n 1 "$COMBINED_FILE" > "$TEST_FILE"
            tail -n +2 "$COMBINED_FILE" | tail -n 50 >> "$TEST_FILE"
        else
            echo "  Not enough data for train/test split (need >50 rows, found $data_lines)"
        fi
        
        echo "Completed processing $attack_name"
        echo "---------------------------------"
    fi
done

echo "All attack folders processed!" 