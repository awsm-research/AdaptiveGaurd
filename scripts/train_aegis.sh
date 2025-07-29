#!/bin/bash
python main_energy_attack.py \
    --do_train \
    --train_data_file aegis_dataset/labeled_first_category/aegis_train_100_or_less.csv \
    --eval_data_file aegis_dataset/labeled_first_category/aegis_validation.csv \
    --test_data_file aegis_dataset/combined_test_validation_less_than_100_RQ1.csv \
    --output_dir gpt2_aegis_model \
    --model_type gpt2 \
    --model_name_or_path gpt2 \
    --tokenizer_name gpt2 \
    --model_name gpt2_aegis_model_filtered.bin \
    --block_size 512 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 1e-4 \
    --_lambda 0.5 \
    --max_grad_norm 2.0 \
    --epochs 10 \
    --seed 42 \
    --evaluate_during_training 2>&1 | tee gpt2_aegis_testing_filtered.log



    