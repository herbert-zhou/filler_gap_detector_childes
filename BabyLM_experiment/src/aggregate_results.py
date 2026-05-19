#!/usr/bin/env python3
"""
Aggregate analysis results from individual CSV files into a single comprehensive dataset.

Usage:
    python aggregate_results.py [--results_dir RESULTS_DIR] [--output OUTPUT_PATH] [--filter_key VALUE]
    
Examples:
    # Aggregate all results
    python aggregate_results.py
    
    # Aggregate results from specific directory
    python aggregate_results.py --results_dir results/par
    
    # Filter by specific dataset filter
    python aggregate_results.py --dataset_filter embedded_questions
    
    # Generate summary statistics
    python aggregate_results.py --summary
"""

import argparse
import pandas as pd
from pathlib import Path
import sys


def find_result_files(results_dir):
    """Recursively find all CSV files in the results directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory does not exist: {results_dir}")
        return []
    
    csv_files = list(results_path.rglob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {results_dir}")
    return csv_files


def aggregate_results(results_dir, output_path=None, filter_params=None):
    """
    Aggregate all individual CSV results into a single DataFrame.
    
    Args:
        results_dir: Path to the results directory
        output_path: Optional path to save aggregated results
        filter_params: Optional dict of parameters to filter by
    
    Returns:
        Aggregated DataFrame
    """
    csv_files = find_result_files(results_dir)
    
    if not csv_files:
        print("No CSV files found!")
        return None
    
    # Read all CSV files
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    if not dfs:
        print("No valid CSV files could be read!")
        return None
    
    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Aggregated {len(dfs)} CSV files into {len(combined_df)} rows")
    
    # Apply filters if provided
    if filter_params:
        for key, value in filter_params.items():
            if key in combined_df.columns:
                combined_df = combined_df[combined_df[key] == value]
                print(f"Filtered by {key}={value}: {len(combined_df)} rows remaining")
    
    # Sort by key columns for readability
    sort_columns = []
    for col in ['include_dir', 'dataset_filter', 'model_config', 'dataset_size', 'seed', 'control', 'minimal_pair_category']:
        if col in combined_df.columns:
            sort_columns.append(col)
    
    if sort_columns:
        combined_df = combined_df.sort_values(sort_columns)
    
    # Save if output path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Saved aggregated results to: {output_file}")
    
    return combined_df


def generate_summary(df, group_by=None):
    """
    Generate summary statistics from aggregated results.
    
    Args:
        df: Aggregated DataFrame
        group_by: List of columns to group by (default: all except proportion_right)
    """
    if df is None or df.empty:
        print("No data to summarize")
        return None
    
    if group_by is None:
        # Default grouping: everything except the metric
        group_by = [col for col in df.columns if col not in ['proportion_right', 'model']]
        if 'checkpoint' in group_by:
            group_by.remove('checkpoint')
    
    # Remove any columns that don't exist
    group_by = [col for col in group_by if col in df.columns]
    
    if not group_by:
        print("No columns to group by")
        return None
    
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    # Overall statistics
    print("Overall Statistics:")
    print(f"  Mean accuracy: {df['proportion_right'].mean():.4f}")
    print(f"  Std accuracy:  {df['proportion_right'].std():.4f}")
    print(f"  Min accuracy:  {df['proportion_right'].min():.4f}")
    print(f"  Max accuracy:  {df['proportion_right'].max():.4f}")
    print(f"  Total rows:    {len(df)}")
    
    # Group statistics
    print(f"\n{'='*80}")
    print("Grouped Statistics:")
    print(f"{'='*80}\n")
    
    summary = df.groupby(group_by)['proportion_right'].agg(['mean', 'std', 'count'])
    summary = summary.sort_values('mean', ascending=False)
    
    return summary


def compare_control_vs_filtered(df):
    """Compare performance between control and filtered models."""
    if df is None or df.empty:
        print("No data to compare")
        return None
    
    if 'control' not in df.columns:
        print("No control column found")
        return None
    
    print(f"\n{'='*80}")
    print("CONTROL VS FILTERED COMPARISON")
    print(f"{'='*80}\n")
    
    # Group by all factors except control and seed
    group_cols = [col for col in ['dataset_filter', 'model_config', 'dataset_size', 
                                    'include_dir', 'minimal_pair_category'] 
                  if col in df.columns]
    
    if not group_cols:
        print("Insufficient columns for comparison")
        return None
    
    # Calculate mean across seeds for each configuration
    comparison = df.groupby(group_cols + ['control'])['proportion_right'].mean().unstack(fill_value=0)
    
    # Add difference column
    if False in comparison.columns and True in comparison.columns:
        comparison['difference'] = comparison[False] - comparison[True]
        comparison = comparison.sort_values('difference', ascending=False)
        
        print(f"Top 10 improvements (filtered > control):")
        print(comparison.head(10))
        print(f"\nTop 10 regressions (control > filtered):")
        print(comparison.tail(10))
    
    return comparison


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate analysis results from individual CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Aggregate all results
    python aggregate_results.py
    
    # Aggregate and save to specific file
    python aggregate_results.py --output aggregated_results.csv
    
    # Filter by dataset filter type
    python aggregate_results.py --dataset_filter embedded_questions
    
    # Generate summary statistics
    python aggregate_results.py --summary
    
    # Compare control vs filtered
    python aggregate_results.py --compare
    
    # Combine multiple options
    python aggregate_results.py --summary --compare --output all_results.csv
        """
    )
    
    parser.add_argument("--results_dir", type=str, default="analysis_data",
                        help="Directory containing result CSV files (default: analysis_data)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for aggregated CSV (default: don't save)")
    
    # Filter options
    parser.add_argument("--dataset_filter", type=str, default=None,
                        help="Filter by dataset filter type")
    parser.add_argument("--model_config", type=str, default=None,
                        help="Filter by model configuration")
    parser.add_argument("--dataset_size", type=str, default=None,
                        help="Filter by dataset size")
    parser.add_argument("--include_dir", type=str, default=None,
                        help="Filter by include directory")
    parser.add_argument("--minimal_pair_category", type=str, default=None,
                        help="Filter by minimal pair category")
    parser.add_argument("--seed", type=int, default=None,
                        help="Filter by seed")
    parser.add_argument("--control", action="store_true",
                        help="Filter for control models only")
    parser.add_argument("--filtered", action="store_true",
                        help="Filter for filtered models only")
    
    # Analysis options
    parser.add_argument("--summary", action="store_true",
                        help="Generate summary statistics")
    parser.add_argument("--compare", action="store_true",
                        help="Compare control vs filtered models")
    parser.add_argument("--group_by", type=str, nargs='+', default=None,
                        help="Columns to group by for summary (space-separated)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Build filter parameters
    filter_params = {}
    if args.dataset_filter:
        filter_params['dataset_filter'] = args.dataset_filter
    if args.model_config:
        filter_params['model_config'] = args.model_config
    if args.dataset_size:
        filter_params['dataset_size'] = args.dataset_size
    if args.include_dir:
        filter_params['include_dir'] = args.include_dir
    if args.minimal_pair_category:
        filter_params['minimal_pair_category'] = args.minimal_pair_category
    if args.seed is not None:
        filter_params['seed'] = args.seed
    if args.control:
        filter_params['control'] = True
    elif args.filtered:
        filter_params['control'] = False
    
    # Aggregate results
    df = aggregate_results(args.results_dir, args.output, filter_params)
    
    if df is None:
        sys.exit(1)
    
    # Generate summary if requested
    if args.summary:
        summary = generate_summary(df, args.group_by)
        if summary is not None:
            print(summary.to_string())
    
    # Compare control vs filtered if requested
    if args.compare:
        comparison = compare_control_vs_filtered(df)
        if comparison is not None:
            print(comparison.to_string())
    
    # Display basic info if no special output requested
    if not args.summary and not args.compare:
        print(f"\nDataFrame shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head(10).to_string())
        print(f"\nColumns: {list(df.columns)}")
        
        if 'minimal_pair_category' in df.columns:
            print(f"\nUnique minimal pair categories: {df['minimal_pair_category'].nunique()}")
            print(df['minimal_pair_category'].value_counts().to_string())


if __name__ == "__main__":
    main()
