import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def calculate_ood_threshold(train_df, metric, quantile):
    """
    Calculate OOD threshold based on specified metric and quantile from training data
    """
    if metric not in train_df.columns:
        raise ValueError(f"Metric '{metric}' not found in training data")
    
    threshold = np.quantile(train_df[metric], quantile/100)
    return threshold

def analyze_ood_detection(train_file, test_file, metrics, quantile, output_dir=None):
    """
    Analyze OOD detection performance using specified metrics and quantile threshold
    
    Parameters:
    -----------
    train_file : str
        Path to training CSV file with metrics for threshold calculation
    test_file : str
        Path to test CSV file with metrics to evaluate
    metrics : list
        List of metrics to use for OOD detection (e.g., ['energy_scores', 'mahalanobis_distances'])
    quantile : float
        Quantile to use for threshold (e.g., 95 for 95th percentile)
    output_dir : str, optional
        Directory to save plots and results
    """
    # Load the data
    try:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Validate that required columns exist
    required_columns = metrics + ['source_type']
    missing_train = [col for col in metrics if col not in train_df.columns]
    missing_test = [col for col in required_columns if col not in test_df.columns]
    
    if missing_train:
        print(f"Error: Missing columns in train file: {missing_train}")
        return
    
    if missing_test:
        print(f"Error: Missing columns in test file: {missing_test}")
        return
    
    # Calculate thresholds for each metric
    thresholds = {}
    for metric in metrics:
        threshold = calculate_ood_threshold(train_df, metric, quantile)
        thresholds[metric] = threshold
        print(f"{metric} threshold ({quantile}th quantile): {threshold:.6f}")
    
    # Apply thresholds to test data
    for metric in metrics:
        # For some metrics like energy_scores, "above" threshold is OOD
        # For others like likelihood_ratios, "below" threshold is OOD
        # Adjust comparison direction based on metric
        if metric in ['energy_scores', 'mahalanobis_distances', 'uncertainties']:
            test_df[f'{metric}_is_ood'] = test_df[metric] > thresholds[metric]
        else:  # likelihood_ratios
            test_df[f'{metric}_is_ood'] = test_df[metric] > thresholds[metric]
    
    # Combine multiple metrics with AND logic if multiple metrics selected
    if len(metrics) > 1:
        combined_column = f"combined_{'_'.join(metrics)}_is_ood"
        test_df[combined_column] = True
        for metric in metrics:
            test_df[combined_column] &= test_df[f'{metric}_is_ood']
    else:
        combined_column = f"{metrics[0]}_is_ood"
    
    # Calculate detection rates
    jailbreak_samples = test_df[test_df['source_type'] == 'jailbreak']
    validation_samples = test_df[test_df['source_type'] == 'validation']
    
    # Overall statistics
    total_samples = len(test_df)
    total_jailbreak = len(jailbreak_samples)
    total_validation = len(validation_samples)
    
    # OOD detection statistics
    ood_detected = test_df[combined_column].sum()
    jailbreak_detected = jailbreak_samples[combined_column].sum()
    validation_falsely_detected = validation_samples[combined_column].sum()
    
    # Calculate percentages
    overall_detection_rate = ood_detected / total_samples * 100
    jailbreak_detection_rate = jailbreak_detected / total_jailbreak * 100 if total_jailbreak > 0 else 0
    false_positive_rate = validation_falsely_detected / total_validation * 100 if total_validation > 0 else 0
    
    # Print results
    print("\n===== OOD Detection Results =====")
    print(f"Metrics used: {', '.join(metrics)}")
    print(f"Quantile threshold: {quantile}%")
    print(f"Total samples: {total_samples}")
    print(f"Jailbreak samples: {total_jailbreak}")
    print(f"Validation samples: {total_validation}")
    print("\nDetection statistics:")
    print(f"Total OOD detected: {ood_detected} ({overall_detection_rate:.2f}%)")
    print(f"Jailbreak samples detected as OOD: {jailbreak_detected} ({jailbreak_detection_rate:.2f}%)")
    print(f"Validation samples falsely detected as OOD: {validation_falsely_detected} ({false_positive_rate:.2f}%)")
    
    # Create visualization
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Plot histogram for each metric
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            
            # Create histograms by source type
            sns.histplot(data=test_df, x=metric, hue='source_type', bins=50, alpha=0.6, element="step")
            
            # Add vertical line for threshold
            plt.axvline(x=thresholds[metric], color='red', linestyle='--', label=f'{quantile}th Quantile Threshold')
            
            plt.title(f'Distribution of {metric} by Source Type')
            plt.xlabel(metric)
            plt.ylabel('Count')
            plt.legend()
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(output_path / f"{metric}_distribution.png", dpi=300)
            plt.close()
        
        # Create a confusion matrix style visualization
        plt.figure(figsize=(10, 8))
        conf_matrix = np.array([
            [total_validation - validation_falsely_detected, validation_falsely_detected],
            [total_jailbreak - jailbreak_detected, jailbreak_detected]
        ])
        
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Not OOD', 'OOD'], 
                   yticklabels=['Validation', 'Jailbreak'])
        
        plt.title(f'OOD Detection Results ({", ".join(metrics)}, {quantile}th Quantile)')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        # Save the confusion matrix
        plt.savefig(output_path / f"ood_detection_matrix_{'_'.join(metrics)}_{quantile}.png", dpi=300)
        plt.close()
        
        # Save results to CSV
        results = {
            'metrics': [', '.join(metrics)],
            'quantile': [quantile],
            'threshold_values': [', '.join([f"{metric}: {thresholds[metric]:.6f}" for metric in metrics])],
            'total_samples': [total_samples],
            'jailbreak_samples': [total_jailbreak],
            'validation_samples': [total_validation],
            'total_ood_detected': [ood_detected],
            'overall_detection_rate': [overall_detection_rate],
            'jailbreak_detection_rate': [jailbreak_detection_rate],
            'false_positive_rate': [false_positive_rate]
        }
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path / f"ood_detection_results_{'_'.join(metrics)}_{quantile}.csv", index=False)
        print(f"Results and plots saved to {output_path}")
    
    return {
        'thresholds': thresholds,
        'jailbreak_detection_rate': jailbreak_detection_rate,
        'false_positive_rate': false_positive_rate,
        'test_df': test_df  # Return the dataframe with OOD predictions for further analysis
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze OOD detection performance')
    parser.add_argument('--train_file', type=str, required=True, 
                        help='Path to training CSV file with metrics for threshold calculation')
    parser.add_argument('--test_file', type=str, required=True, 
                        help='Path to test CSV file with metrics to evaluate')
    parser.add_argument('--metrics', type=str, nargs='+', required=True, 
                        choices=['energy_scores', 'mahalanobis_distances', 'likelihood_ratios', 'uncertainties'],
                        help='Metrics to use for OOD detection')
    parser.add_argument('--quantile', type=float, default=95, 
                        help='Quantile to use for threshold (e.g., 95 for 95th percentile)')
    parser.add_argument('--output_dir', type=str, default='ood_detection_results', 
                        help='Directory to save plots and results')
    
    args = parser.parse_args()
    
    analyze_ood_detection(
        args.train_file, 
        args.test_file, 
        args.metrics, 
        args.quantile, 
        args.output_dir
    )

if __name__ == '__main__':
    main() 