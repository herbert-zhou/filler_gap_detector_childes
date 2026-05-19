import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.patches as mpatches
import seaborn as sns 
from scipy import stats

plt.rcParams['figure.titlesize'] = 18
plt.rcParams["figure.titleweight"] = "bold"

# Base directory for analysis results
ANALYSIS_DATA_DIR = "./analysis_data"


def load_all_categories_for_seed(include_dir, dataset_filter, model_config, dataset_size, seed, lr_wd=None, control=False):
    """Load all minimal pair category results for a specific seed.
    
    Returns a DataFrame with columns: sentence_type, proportion_right
    """
    control_suffix = "-control" if control else ""
    if lr_wd:
        seed_dir = Path(ANALYSIS_DATA_DIR, include_dir, dataset_filter, 
                        f"{model_config}-{dataset_size}{control_suffix}", lr_wd, f"seed_{seed}")
    else:
        seed_dir = Path(ANALYSIS_DATA_DIR, include_dir, dataset_filter, 
                        f"{model_config}-{dataset_size}{control_suffix}", f"seed_{seed}")
    
    if not seed_dir.exists():
        return None
    
    # Find all CSV files in the seed directory
    csv_files = list(seed_dir.glob("*.csv"))
    if not csv_files:
        return None
    
    # Read all category CSVs and combine
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Each CSV should have minimal_pair_category and proportion_right
        if 'minimal_pair_category' in df.columns and 'proportion_right' in df.columns:
            for _, row in df.iterrows():
                all_data.append({
                    'sentence_type': row['minimal_pair_category'],
                    'proportion_right': row['proportion_right']
                })
    
    if not all_data:
        return None
    
    return pd.DataFrame(all_data)

def plot_double_bar_plot(include_dir, dataset_filter, model_config, dataset_size, seed, save_path, lr_wd=None):
    """Plot comparison between filtered and control models for a single seed.
    
    Now uses the new analysis_data directory structure.
    """
    # Load data from analysis_data directory
    df1 = load_all_categories_for_seed(include_dir, dataset_filter, model_config, dataset_size, seed, lr_wd=lr_wd, control=False)
    df2 = load_all_categories_for_seed(include_dir, dataset_filter, model_config, dataset_size, seed, lr_wd=lr_wd, control=True)
    
    if df1 is None or df2 is None:
        print(f"Could not load data for {include_dir}/{dataset_filter}/{model_config}-{dataset_size}/seed_{seed}")
        return
    
    if len(df1) == 0 or len(df2) == 0:
        print(f"Empty data for {include_dir}/{dataset_filter}/{model_config}-{dataset_size}/seed_{seed}")
        return
    
    label1 = f"{dataset_filter}_{model_config}-{dataset_size}"
    label2 = f"{dataset_filter}_{model_config}-{dataset_size}-control"
    model = f"{model_config}-{dataset_size}"
    sentence_type = dataset_filter

    # Merge dataframes to ensure alignment
    df1_indexed = df1.set_index('sentence_type')
    df2_indexed = df2.set_index('sentence_type')
    
    # Find common categories
    common_categories = df1_indexed.index.intersection(df2_indexed.index)
    if len(common_categories) == 0:
        print(f"No common categories for {include_dir}/{dataset_filter}/{model_config}-{dataset_size}/seed_{seed}")
        return
    
    categories = common_categories
    values1 = df1_indexed.loc[common_categories, 'proportion_right']
    values2 = df2_indexed.loc[common_categories, 'proportion_right']
    n_categories = len(categories)
    n_cols = min(n_categories, 4)
    n_rows = (n_categories + n_cols - 1) // n_cols
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axs = axs.flatten() if n_categories > 1 else [axs]
    color1 = 'xkcd:sky blue'
    color2 = 'xkcd:eggshell'

    patch1 = mpatches.Patch(color=color1, label=model)
    patch2 = mpatches.Patch(color=color2, label=f'{model}-control')

    for i, cat in enumerate(categories):
        bar_width = 0.35  # Width of each bar
        axs[i].bar(1 - bar_width/2, values1.loc[cat], bar_width, color=color1)
        axs[i].bar(1 + bar_width/2, values2.loc[cat], bar_width, color=color2)
        axs[i].set_title(cat)
        axs[i].tick_params(axis='x', length=0, labelbottom=False) 
        axs[i].set_ylim((0, 1))
    # Hide any unused subplots
    for i in range(n_categories, len(axs)):
        axs[i].set_visible(False)
    sentence_type_str = " ".join(sentence_type.split("_"))
    fig.suptitle(f'Accuracy comparison between {model} with {sentence_type_str} filtered out and its control')
    plt.legend(bbox_to_anchor=(0,0), handles=[patch1, patch2])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_double_bar_plot_seeds(include_dir, dataset_filter, model_config, dataset_size, save_path, lr_wd=None):
    """Plot comparison averaged across all seeds.
    
    Now uses the new analysis_data directory structure.
    """
    # Determine available seeds
    base_dir = Path(ANALYSIS_DATA_DIR, include_dir, dataset_filter, f"{model_config}-{dataset_size}")
    if lr_wd:
        base_dir = base_dir / lr_wd
    
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return
    
    seeds = sorted([int(d.name.split('_')[1]) for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('seed_')])
    
    if not seeds:
        print(f"No seeds found in {base_dir}")
        return

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    
    for seed in seeds:
        seed_df1 = load_all_categories_for_seed(include_dir, dataset_filter, model_config, dataset_size, seed, lr_wd=lr_wd, control=False)
        seed_df2 = load_all_categories_for_seed(include_dir, dataset_filter, model_config, dataset_size, seed, lr_wd=lr_wd, control=True)
        
        if seed_df1 is not None:
            df1 = pd.concat([df1, seed_df1], axis=0)
        if seed_df2 is not None:
            df2 = pd.concat([df2, seed_df2], axis=0)

    if df1.empty or df2.empty:
        print(f"No data found for {include_dir}/{dataset_filter}/{model_config}-{dataset_size}")
        return

    mean_data_1 = df1.groupby('sentence_type')['proportion_right'].mean()
    std_data_1 = df1.groupby('sentence_type')['proportion_right'].std()

    mean_data_2 = df2.groupby('sentence_type')['proportion_right'].mean()
    std_data_2 = df2.groupby('sentence_type')['proportion_right'].std()

    # Find common categories between both datasets
    common_categories = mean_data_1.index.intersection(mean_data_2.index)
    if len(common_categories) == 0:
        print(f"No common categories for {include_dir}/{dataset_filter}/{model_config}-{dataset_size}")
        return
    
    categories = common_categories.tolist()
    n_categories = len(categories)
    n_cols = min(n_categories, 4)
    n_rows = (n_categories + n_cols - 1) // n_cols
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axs = axs.flatten() if n_categories > 1 else [axs]
    color1 = 'xkcd:sky blue'
    color2 = 'xkcd:eggshell'

    model = f"{model_config}-{dataset_size}"
    patch1 = mpatches.Patch(color=color1, label=model)
    patch2 = mpatches.Patch(color=color2, label=f'{model}-control')

    for i, cat in enumerate(categories):
        bar_width = 0.35  # Width of each bar
        # Handle NaN in std (happens when only one value exists)
        std1 = std_data_1.loc[cat] if not pd.isna(std_data_1.loc[cat]) else 0
        std2 = std_data_2.loc[cat] if not pd.isna(std_data_2.loc[cat]) else 0
        
        axs[i].bar(1 - bar_width/2, mean_data_1.loc[cat], bar_width, yerr=std1, color=color1,
            ecolor='red', capsize=1.5)
        axs[i].bar(1 + bar_width/2, mean_data_2.loc[cat], bar_width, yerr=std2, color=color2,
            ecolor='red', capsize=1.5)
        axs[i].set_title(cat)
        axs[i].tick_params(axis='x', length=0, labelbottom=False) 
        axs[i].set_ylim((0, 1))
    # Hide any unused subplots
    for i in range(n_categories, len(axs)):
        axs[i].set_visible(False)

    sentence_type = dataset_filter
    sentence_type_str = " ".join(sentence_type.split("_"))
    fig.suptitle(f'Accuracy comparison between {model} with {sentence_type_str} filtered out and its control')
    plt.legend(bbox_to_anchor=(0,0), handles=[patch1, patch2])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
# def plot_heatmap(csv_path_1, csv_path_2, save_path):
#     df1 = pd.read_csv(csv_path_1)
#     label1 = csv_path_1.parent.parent.name + "_" + csv_path_1.parent.name
    
#     model = csv_path_1.parent.name
#     sentence_type = csv_path_1.parent.parent.name

#     df2 = pd.read_csv(csv_path_2)
#     label2 = csv_path_2.parent.parent.name + "_" + csv_path_2.parent.name

#     categories = df1['sentence_type']
#     values1 = df1['proportion_right']
#     values2 = df2['proportion_right']
#     differences = values1 - values2

#     plt.bar(x=categories, y=differences, ylim=(-1,1))

#     fig.suptitle(f'Differences for {model} on {sentence_type}')
#     plt.legend(bbox_to_anchor=(0,0), handles=[patch1, patch2])

#     plt.savefig(save_path)

def discover_analysis_data():
    """Discover all available configurations in analysis_data directory.
    
    Returns: List of tuples (include_dir, dataset_filter, model_config, dataset_size, lr_wd)
    """
    configs = []
    analysis_path = Path(ANALYSIS_DATA_DIR)
    
    if not analysis_path.exists():
        print(f"Analysis data directory not found: {ANALYSIS_DATA_DIR}")
        return configs
    
    # Scan for include_dirs
    for include_dir in analysis_path.iterdir():
        if not include_dir.is_dir():
            continue
            
        # Scan for dataset_filters
        for dataset_filter_dir in include_dir.iterdir():
            if not dataset_filter_dir.is_dir():
                continue
            dataset_filter = dataset_filter_dir.name
            
            # Scan for model configurations
            for model_dir in dataset_filter_dir.iterdir():
                if not model_dir.is_dir() or model_dir.name.endswith('-control'):
                    continue
                
                # Parse model_config-size from directory name
                model_full = model_dir.name
                # Handle formats like "llama-360M-10M" or "gpt-705M-10M"
                parts = model_full.rsplit('-', 1)
                if len(parts) == 2:
                    model_config = parts[0]
                    dataset_size = parts[1]
                    
                    # Check for lr/wd subdirectories
                    lr_wd_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('lr')]
                    
                    if lr_wd_dirs:
                        # Add config for each lr/wd combination
                        for lr_wd_dir in lr_wd_dirs:
                            configs.append((include_dir.name, dataset_filter, model_config, dataset_size, lr_wd_dir.name))
                    else:
                        # Old structure 
                        configs.append((include_dir.name, dataset_filter, model_config, dataset_size, None))
    
    return configs

def plot_model_and_control():
    """Plot comparison between filtered and control models for individual seeds.
    
    Now uses the new analysis_data directory structure.
    """
    BASE_FIGURES_DIR = "./figures"
    configs = discover_analysis_data()
    
    if not configs:
        print("No analysis data found to plot")
        return
    
    print(f"Found {len(configs)} configurations to plot")
    for include_dir, sentence_filter, model_type, size, lr_wd in configs:
        # Find first available seed
        base_dir = Path(ANALYSIS_DATA_DIR, include_dir, sentence_filter, f"{model_type}-{size}")
        if lr_wd:
            base_dir = base_dir / lr_wd
        
        if not base_dir.exists():
            print(f"Skipping {include_dir}/{sentence_filter}/{model_type}-{size}/{lr_wd or ''} - no data found")
            continue
        
        seeds = sorted([int(d.name.split('_')[1]) for d in base_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('seed_')])
        
        if not seeds:
            print(f"No seeds found for {include_dir}/{sentence_filter}/{model_type}-{size}/{lr_wd or ''}")
            continue
        
        # Use first available seed
        seed = seeds[0]
        filename_suffix = f"-{lr_wd}" if lr_wd else ""
        save_path = Path(BASE_FIGURES_DIR, "double_bar_plots", f"{include_dir}-{sentence_filter}-{model_type}-{size}{filename_suffix}.png")
        print(f"Plotting {include_dir}/{sentence_filter}/{model_type}-{size}/{lr_wd or ''} (seed {seed})")
        plot_double_bar_plot(include_dir, sentence_filter, model_type, size, seed, save_path, lr_wd=lr_wd)


def plot_model_and_control_seeds():
    """Plot comparison between filtered and control models averaged across seeds.
    
    Now uses the new analysis_data directory structure.
    """
    BASE_FIGURES_DIR = "./figures"
    configs = discover_analysis_data()
    
    if not configs:
        print("No analysis data found to plot")
        return
    
    print(f"Found {len(configs)} configurations to plot with seed aggregation")
    for include_dir, sentence_filter, model_type, size, lr_wd in configs:
        filename_suffix = f"-{lr_wd}" if lr_wd else ""
        save_path = Path(BASE_FIGURES_DIR, "double_bar_plots", f"seeds-aggregate-{include_dir}-{sentence_filter}-{model_type}-{size}{filename_suffix}.png")
        print(f"Processing {include_dir}/{sentence_filter}/{model_type}-{size}/{lr_wd or ''}")
        plot_double_bar_plot_seeds(include_dir, sentence_filter, model_type, size, save_path, lr_wd=lr_wd)

def get_checkpoint(checkpoint_str):
    return checkpoint_str[len("checkpoint-"):]


def compute_paired_statistics(filtered_values, control_values):
    """Compute paired statistical tests and effect size for matched seed values."""
    filtered_arr = np.asarray(filtered_values, dtype=float)
    control_arr = np.asarray(control_values, dtype=float)

    n_pairs = min(len(filtered_arr), len(control_arr))
    if n_pairs == 0:
        return {
            'n_pairs': 0,
            'mean_filtered': np.nan,
            'mean_control': np.nan,
            'std_filtered': np.nan,
            'std_control': np.nan,
            'mean_diff': np.nan,
            'cohens_d': np.nan,
            'ttest_p': np.nan,
            'wilcoxon_p': np.nan,
            'selected_test': 'none',
            'selected_p': np.nan,
        }

    filtered_arr = filtered_arr[:n_pairs]
    control_arr = control_arr[:n_pairs]
    diffs = filtered_arr - control_arr

    mean_diff = np.mean(diffs)
    diff_std = np.std(diffs, ddof=1) if n_pairs > 1 else 0.0
    cohens_d = mean_diff / diff_std if diff_std > 0 else np.nan

    ttest_p = np.nan
    if n_pairs > 1 and not np.allclose(diffs, diffs[0]):
        try:
            _, ttest_p = stats.ttest_rel(filtered_arr, control_arr)
        except Exception:
            ttest_p = np.nan

    wilcoxon_p = np.nan
    if n_pairs >= 3 and not np.allclose(diffs, 0):
        try:
            _, wilcoxon_p = stats.wilcoxon(filtered_arr, control_arr, zero_method='wilcox', alternative='two-sided', mode='auto')
        except Exception:
            wilcoxon_p = np.nan

    if n_pairs < 10 and not np.isnan(wilcoxon_p):
        selected_test = 'wilcoxon'
        selected_p = wilcoxon_p
    else:
        selected_test = 'paired_ttest'
        selected_p = ttest_p

    return {
        'n_pairs': n_pairs,
        'mean_filtered': np.mean(filtered_arr),
        'mean_control': np.mean(control_arr),
        'std_filtered': np.std(filtered_arr, ddof=1) if n_pairs > 1 else 0.0,
        'std_control': np.std(control_arr, ddof=1) if n_pairs > 1 else 0.0,
        'mean_diff': mean_diff,
        'cohens_d': cohens_d,
        'ttest_p': ttest_p,
        'wilcoxon_p': wilcoxon_p,
        'selected_test': selected_test,
        'selected_p': selected_p,
    }


def bonferroni_correct(p_values):
    """Simple Bonferroni correction with NaN handling."""
    valid_count = sum(1 for p in p_values if not pd.isna(p))
    corrected = []
    for p in p_values:
        if pd.isna(p) or valid_count == 0:
            corrected.append(np.nan)
        else:
            corrected.append(min(p * valid_count, 1.0))
    return corrected




def plot_checkpoints():
    """Plot heatmaps across seeds.
    
    Now uses the new analysis_data directory structure.
    """
    BASE_FIGURES_DIR = "./figures"
    configs = discover_analysis_data()
    
    if not configs:
        print("No analysis data found to plot")
        return
    
    print(f"Found {len(configs)} configurations for heatmap plotting")
    for include_dir, dataset_filter, model_config, dataset_size, lr_wd in configs:
        base_dir = Path(ANALYSIS_DATA_DIR, include_dir, dataset_filter, f"{model_config}-{dataset_size}")
        if lr_wd:
            base_dir = base_dir / lr_wd
        
        if not base_dir.exists():
            print(f"Skipping {base_dir} - directory not found")
            continue
        
        # Get available seeds
        seeds = sorted([int(d.name.split('_')[1]) for d in base_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('seed_')])
        
        if not seeds:
            print(f"Skipping {base_dir} - no seeds found")
            continue
        
        data1 = []
        data2 = []
        all_categories = set()
        
        for seed in seeds:
            df1 = load_all_categories_for_seed(include_dir, dataset_filter, model_config, dataset_size, seed, control=False)
            df2 = load_all_categories_for_seed(include_dir, dataset_filter, model_config, dataset_size, seed, control=True)
            
            if df1 is not None:
                values1 = df1.set_index('sentence_type')['proportion_right']
                data1.append(values1)
                all_categories.update(values1.index.tolist())
            
            if df2 is not None:
                values2 = df2.set_index('sentence_type')['proportion_right']
                data2.append(values2)
        
        if not data1 or not all_categories:
            continue
        
        categories = sorted(list(all_categories))
        final_df = pd.DataFrame(data1)
        final_df_control = pd.DataFrame(data2)
        
        # Reorder columns to match categories and fill missing values
        final_df = final_df.reindex(columns=categories)
        final_df_control = final_df_control.reindex(columns=categories)
        
        # Plot filtered model heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(final_df, yticklabels=[f"seed_{s}" for s in seeds[:len(data1)]], xticklabels=categories)
        plt.xticks(rotation=45, ha='right')
        filename_suffix = f"-{lr_wd}" if lr_wd else ""
        save_path = Path(BASE_FIGURES_DIR, "heatmaps", f"{include_dir}-{dataset_filter}-{model_config}-{dataset_size}{filename_suffix}.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        title_suffix = f" ({lr_wd})" if lr_wd else ""
        plt.suptitle(f"Heatmap of Accuracy on Minimal Pair Tests across Seeds for {include_dir}-{dataset_filter}-{model_config}-{dataset_size}{title_suffix}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        # Plot control model heatmap
        plt.figure(figsize=(14, 8))
        plt.suptitle(f"Heatmap of Accuracy on Minimal Pair Tests across Seeds for {include_dir}-{dataset_filter}-{model_config}-{dataset_size}_control{title_suffix}")
        sns.heatmap(final_df_control, yticklabels=[f"seed_{s}" for s in seeds[:len(data2)]], xticklabels=categories)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_path = Path(BASE_FIGURES_DIR, "heatmaps", f"{include_dir}-{dataset_filter}-{model_config}-{dataset_size}{filename_suffix}_control.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()

def plot_checkpoint_correlations():
    """Plot Spearman correlations of minimal pair evaluations across seeds.
    
    Now uses the new analysis_data directory structure.
    """
    BASE_FIGURES_DIR = "./figures"
    configs = discover_analysis_data()
    
    if not configs:
        print("No analysis data found to plot")
        return
    
    print(f"Found {len(configs)} configurations for correlation plotting")
    for include_dir, dataset_filter, model_config, dataset_size, lr_wd in configs:
        base_dir = Path(ANALYSIS_DATA_DIR, include_dir, dataset_filter, f"{model_config}-{dataset_size}")
        if lr_wd:
            base_dir = base_dir / lr_wd
        
        if not base_dir.exists():
            print(f"Skipping {base_dir} - directory not found")
            continue
        
        # Get available seeds
        seeds = sorted([int(d.name.split('_')[1]) for d in base_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('seed_')])
        
        if not seeds:
            print(f"Skipping {base_dir} - no seeds found")
            continue
        
        data1 = []
        data2 = []
        all_categories = set()
        
        for seed in seeds:
            df1 = load_all_categories_for_seed(include_dir, dataset_filter, model_config, dataset_size, seed, control=False)
            df2 = load_all_categories_for_seed(include_dir, dataset_filter, model_config, dataset_size, seed, control=True)
            
            if df1 is not None:
                values1 = df1.set_index('sentence_type')['proportion_right']
                data1.append(values1)
                all_categories.update(values1.index.tolist())
            
            if df2 is not None:
                values2 = df2.set_index('sentence_type')['proportion_right']
                data2.append(values2)

        if not data1 or not all_categories:
            continue
        
        categories = sorted(list(all_categories))
        final_df = pd.DataFrame(data1)
        final_df_control = pd.DataFrame(data2)
        
        # Reorder columns to match categories and fill missing values
        final_df = final_df.reindex(columns=categories)
        final_df_control = final_df_control.reindex(columns=categories)
        
        # Calculate correlations
        correlation_matrix_1 = final_df.corr(method='spearman')
        correlation_matrix_2 = final_df_control.corr(method='spearman')
        
        # Plot filtered model correlations
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix_1, yticklabels=categories, xticklabels=categories, annot=True, fmt='.2f')
        plt.xticks(rotation=45, ha='right')
        filename_suffix = f"-{lr_wd}" if lr_wd else ""
        save_path = Path(BASE_FIGURES_DIR, "heatmaps", f"{include_dir}-{dataset_filter}-{model_config}-{dataset_size}{filename_suffix}_correlation.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        title_suffix = f" ({lr_wd})" if lr_wd else ""
        plt.suptitle(f"Seed Spearman correlations of minimal pair evaluations for {include_dir}-{dataset_filter}-{model_config}-{dataset_size}{title_suffix}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        # Plot control model correlations
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix_2, yticklabels=categories, xticklabels=categories, annot=True, fmt='.2f')
        plt.xticks(rotation=45, ha='right')
        save_path = Path(BASE_FIGURES_DIR, "heatmaps", f"{include_dir}-{dataset_filter}-{model_config}-{dataset_size}{filename_suffix}_correlation_control.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.suptitle(f"Seed Spearman correlations of minimal pair evaluations for {include_dir}-{dataset_filter}-{model_config}-{dataset_size}_control{title_suffix}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def plot_ablation_specific_evaluations(annotate=True):
    """Plot bar plots showing how models perform on the specific constructions they were trained without.
    
    Creates plots for:
    - Matrix question evaluations for matrix_questions filtered models
    - Relative clause evaluations for relative_clauses filtered models  
    - Embedded question evaluations for embedded_questions filtered models
    """
    BASE_FIGURES_DIR = "./figures"
    
    # Mapping from filter type to the search string for matching categories
    filter_to_search_string = {
        'matrix_questions': 'matrix',
        'relative_clauses': 'relative',
        'embedded_questions': 'embedded'
    }
    
    configs = discover_analysis_data()
    
    # Group by filter type
    filter_configs = {}
    for include_dir, dataset_filter, model_config, dataset_size, lr_wd in configs:
        if dataset_filter in filter_to_search_string:
            if dataset_filter not in filter_configs:
                filter_configs[dataset_filter] = []
            filter_configs[dataset_filter].append((include_dir, model_config, dataset_size, lr_wd))
    
    for filter_type, search_string in filter_to_search_string.items():
        if filter_type not in filter_configs:
            print(f"No configurations found for filter type: {filter_type}")
            continue
            
        # For each configuration with this filter type
        for include_dir, model_config, dataset_size, lr_wd in filter_configs[filter_type]:
            # Collect data across seeds
            base_dir = Path(ANALYSIS_DATA_DIR, include_dir, filter_type, f"{model_config}-{dataset_size}")
            if lr_wd:
                base_dir = base_dir / lr_wd
            
            if not base_dir.exists():
                print(f"Directory not found: {base_dir}")
                continue
            
            seeds = sorted([int(d.name.split('_')[1]) for d in base_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('seed_')])
            
            if not seeds:
                print(f"No seeds found in {base_dir}")
                continue
            
            # Collect data for filtered and control models
            filtered_data = []
            control_data = []
            
            for seed in seeds:
                df_filtered = load_all_categories_for_seed(include_dir, filter_type, model_config, dataset_size, seed, lr_wd=lr_wd, control=False)
                df_control = load_all_categories_for_seed(include_dir, filter_type, model_config, dataset_size, seed, lr_wd=lr_wd, control=True)
                
                if df_filtered is not None:
                    filtered_data.append(df_filtered)
                if df_control is not None:
                    control_data.append(df_control)
            
            if not filtered_data or not control_data:
                print(f"No data found for {filter_type}/{model_config}-{dataset_size}")
                continue
            
            # Combine all seeds
            all_filtered = pd.concat(filtered_data, axis=0)
            all_control = pd.concat(control_data, axis=0)
            
            # Find which category names actually exist in the data
            all_categories = set(all_filtered['sentence_type'].unique())
            matching_categories = [cat for cat in all_categories 
                                  if search_string in cat.lower()]
            
            if not matching_categories:
                print(f"No matching categories found for {filter_type} in {model_config}-{dataset_size}")
                print(f"  Available categories: {sorted(all_categories)}")
                continue
            
            # Build paired seed-level values per category for valid paired tests
            paired_by_category = {cat: {'filtered': [], 'control': []} for cat in matching_categories}
            
            for seed in seeds:
                df_filtered = load_all_categories_for_seed(include_dir, filter_type, model_config, dataset_size, seed, lr_wd=lr_wd, control=False)
                df_control = load_all_categories_for_seed(include_dir, filter_type, model_config, dataset_size, seed, lr_wd=lr_wd, control=True)
                if df_filtered is None or df_control is None:
                    continue

                filtered_series = df_filtered.set_index('sentence_type')['proportion_right']
                control_series = df_control.set_index('sentence_type')['proportion_right']

                for cat in matching_categories:
                    if cat in filtered_series.index and cat in control_series.index:
                        paired_by_category[cat]['filtered'].append(float(filtered_series.loc[cat]))
                        paired_by_category[cat]['control'].append(float(control_series.loc[cat]))

            categories = sorted([
                cat for cat in matching_categories
                if len(paired_by_category[cat]['filtered']) > 0 and len(paired_by_category[cat]['control']) > 0
            ])

            if not categories:
                print(f"No paired category data for {filter_type}/{model_config}-{dataset_size}")
                continue

            stats_by_category = {
                cat: compute_paired_statistics(paired_by_category[cat]['filtered'], paired_by_category[cat]['control'])
                for cat in categories
            }
            raw_p_values = [stats_by_category[cat]['selected_p'] for cat in categories]
            adj_p_values = bonferroni_correct(raw_p_values)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(categories))
            width = 0.35
            
            color_filtered = 'xkcd:sky blue'
            color_control = 'xkcd:eggshell'
            
            ax.bar(
                x - width/2,
                [stats_by_category[cat]['mean_filtered'] for cat in categories],
                width,
                yerr=[
                    stats_by_category[cat]['std_filtered']
                    if not pd.isna(stats_by_category[cat]['std_filtered']) else 0
                    for cat in categories
                ],
                label=f'{model_config}-{dataset_size} (filtered)',
                color=color_filtered,
                capsize=5,
                ecolor='red'
            )
            ax.bar(
                x + width/2,
                [stats_by_category[cat]['mean_control'] for cat in categories],
                width,
                yerr=[
                    stats_by_category[cat]['std_control']
                    if not pd.isna(stats_by_category[cat]['std_control']) else 0
                    for cat in categories
                ],
                label=f'{model_config}-{dataset_size} (control)',
                color=color_control,
                capsize=5,
                ecolor='red'
            )

            if annotate:
                for i, cat in enumerate(categories):
                    stat_result = stats_by_category[cat]
                    p_adj = adj_p_values[i]
                    d_val = stat_result['cohens_d']
                    n_pairs = stat_result['n_pairs']

                    top_val = max(stat_result['mean_filtered'], stat_result['mean_control'])
                    annotation_y = min(0.98, top_val + 0.08)
                    d_text = f"d={d_val:.2f}" if not pd.isna(d_val) else "d=n/a"
                    p_text = f"p={p_adj:.3g}" if not pd.isna(p_adj) else "p=n/a"
                    ax.text(i, annotation_y, f"{p_text}\n{d_text}\nn={n_pairs}",
                            ha='center', va='bottom', fontsize=8)
            
            ax.set_ylabel('Proportion Correct', fontsize=12)
            filter_display = filter_type.replace("_", " ").title()
            ax.set_title(f'{filter_display} Evaluation\nModel trained WITHOUT {filter_display.lower()} vs Control', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([cat.replace('_', ' ') for cat in categories], rotation=45, ha='right')
            ax.legend(fontsize=10)
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            filename_suffix = f"-{lr_wd}" if lr_wd else ""
            output_subdir = "ablation_specific" if annotate else "ablation_specific_no_annotation"
            file_suffix = "" if annotate else "-no-annotation"
            save_path = Path(BASE_FIGURES_DIR, output_subdir,
                           f"{include_dir}-{filter_type}-{model_config}-{dataset_size}{filename_suffix}{file_suffix}.png")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            variant = "annotated" if annotate else "no-annotation"
            print(f"Generated ablation-specific ({variant}) plot: {save_path}")

def plot_ablation_grid(annotate=True):
    """Plot a 3x3 grid showing all ablation datasets vs all aggregate evaluation categories.
    
    Rows: ablation datasets (matrix_questions, relative_clauses, embedded_questions)
    Columns: evaluation categories (embedded, relative, matrix)
    Each subplot: 2 bars (filtered model vs control)
    """
    BASE_FIGURES_DIR = "./figures"
    
    # Define the ablation types and evaluation categories
    ablation_types = ['matrix_questions', 'relative_clauses', 'embedded_questions']
    eval_categories = ['embedded', 'relative', 'matrix']
    
    configs = discover_analysis_data()
    
    # Group configurations by ablation type
    ablation_configs = {abl: [] for abl in ablation_types}
    for include_dir, dataset_filter, model_config, dataset_size, lr_wd in configs:
        if dataset_filter in ablation_types:
            ablation_configs[dataset_filter].append((include_dir, model_config, dataset_size, lr_wd))
    
    # For each unique model configuration, create a 3x3 grid
    all_model_configs = set()
    for abl in ablation_types:
        for include_dir, model_config, dataset_size, lr_wd in ablation_configs[abl]:
            all_model_configs.add((include_dir, model_config, dataset_size, lr_wd))
    
    for include_dir, model_config, dataset_size, lr_wd in all_model_configs:
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle(f'Ablation Grid: {model_config}-{dataset_size}\n' +
                    f'Rows=Training Data Ablation, Columns=Evaluation Category',
                    fontsize=16, fontweight='bold')
        
        data_found = False
        
        for row_idx, ablation_type in enumerate(ablation_types):
            for col_idx, eval_cat in enumerate(eval_categories):
                ax = axes[row_idx, col_idx]
                
                # Try to load data for this ablation type
                base_dir = Path(ANALYSIS_DATA_DIR, include_dir, ablation_type, f"{model_config}-{dataset_size}")
                if lr_wd:
                    base_dir = base_dir / lr_wd
                
                if not base_dir.exists():
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_ylim(0, 1)
                    if row_idx == 2:
                        ax.set_xlabel(eval_cat.replace('_', ' ').title())
                    if col_idx == 0:
                        ax.set_ylabel(ablation_type.replace('_', '\n'))
                    continue
                
                # Get seeds
                seeds = sorted([int(d.name.split('_')[1]) for d in base_dir.iterdir() 
                               if d.is_dir() and d.name.startswith('seed_')])
                
                if not seeds:
                    ax.text(0.5, 0.5, 'No seeds', ha='center', va='center', transform=ax.transAxes)
                    ax.set_ylim(0, 1)
                    if row_idx == 2:
                        ax.set_xlabel(eval_cat.replace('_', ' ').title())
                    if col_idx == 0:
                        ax.set_ylabel(ablation_type.replace('_', '\n'))
                    continue
                
                # Collect matched paired data across seeds for this specific evaluation category
                filtered_values = []
                control_values = []
                
                for seed in seeds:
                    df_filtered = load_all_categories_for_seed(include_dir, ablation_type, model_config, 
                                                               dataset_size, seed, lr_wd=lr_wd, control=False)
                    df_control = load_all_categories_for_seed(include_dir, ablation_type, model_config, 
                                                              dataset_size, seed, lr_wd=lr_wd, control=True)
                    
                    # Find the exact aggregate category name
                    if df_filtered is not None and df_control is not None:
                        # Look for exact aggregate match first, then case-insensitive exact match
                        filtered_matches = df_filtered[df_filtered['sentence_type'] == eval_cat]
                        if filtered_matches.empty:
                            filtered_matches = df_filtered[df_filtered['sentence_type'].str.lower() == eval_cat.lower()]

                        control_matches = df_control[df_control['sentence_type'] == eval_cat]
                        if control_matches.empty:
                            control_matches = df_control[df_control['sentence_type'].str.lower() == eval_cat.lower()]

                        if not filtered_matches.empty and not control_matches.empty:
                            filtered_values.append(float(filtered_matches.iloc[0]['proportion_right']))
                            control_values.append(float(control_matches.iloc[0]['proportion_right']))
                
                # Plot if we have data
                if filtered_values and control_values:
                    stat_result = compute_paired_statistics(filtered_values, control_values)
                    mean_filtered = stat_result['mean_filtered']
                    std_filtered = stat_result['std_filtered'] if not pd.isna(stat_result['std_filtered']) else 0
                    mean_control = stat_result['mean_control']
                    std_control = stat_result['std_control'] if not pd.isna(stat_result['std_control']) else 0
                    
                    x = np.array([0, 1])
                    width = 0.6
                    
                    color_filtered = 'xkcd:sky blue'
                    color_control = 'xkcd:eggshell'
                    
                    ax.bar(x[0], mean_filtered, width, yerr=std_filtered,
                          color=color_filtered, capsize=4, ecolor='red')
                    ax.bar(x[1], mean_control, width, yerr=std_control,
                          color=color_control, capsize=4, ecolor='red')
                    
                    ax.set_ylim(0, 1)
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(['Filtered', 'Control'], fontsize=8)
                    ax.grid(axis='y', alpha=0.3)

                    if annotate:
                        p_val = stat_result['selected_p']
                        d_val = stat_result['cohens_d']
                        d_text = f"d={d_val:.2f}" if not pd.isna(d_val) else "d=n/a"
                        p_text = f"p={p_val:.3g}" if not pd.isna(p_val) else "p=n/a"
                        ax.text(0.5, 0.95, f"{p_text}\n{d_text} n={stat_result['n_pairs']}",
                                ha='center', va='top', transform=ax.transAxes, fontsize=8)
                    
                    data_found = True
                else:
                    ax.text(0.5, 0.5, 'No matching\ncategory', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=9)
                    ax.set_ylim(0, 1)
                
                # Labels
                if row_idx == 2:
                    ax.set_xlabel(eval_cat.replace('_', ' ').title(), fontsize=14)
                if col_idx == 0:
                    ax.set_ylabel(ablation_type.replace('_', '\n'), fontsize=14)
        
        # Only save if we found some data
        if data_found:
            plt.tight_layout()
            filename_suffix = f"-{lr_wd}" if lr_wd else ""
            output_subdir = "ablation_grid" if annotate else "ablation_grid_no_annotation"
            file_suffix = "" if annotate else "-no-annotation"
            save_path = Path(BASE_FIGURES_DIR, output_subdir,
                           f"{include_dir}-grid-{model_config}-{dataset_size}{filename_suffix}{file_suffix}.png")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            variant = "annotated" if annotate else "no-annotation"
            print(f"Generated ablation grid ({variant}): {save_path}")
        else:
            plt.close()
            print(f"Skipped ablation grid for {include_dir}/{model_config}-{dataset_size} - no data found")

if __name__ == "__main__":
    print("Discovering available analysis data...")
    configs = discover_analysis_data()
    print(f"\nFound {len(configs)} configurations:")
    for include_dir, dataset_filter, model_config, dataset_size, lr_wd in configs:
        lr_wd_str = f"/{lr_wd}" if lr_wd else ""
        print(f"  - {include_dir}/{dataset_filter}/{model_config}-{dataset_size}{lr_wd_str}")
    
    print("\nGenerating all figures...")
    plot_model_and_control()
    plot_model_and_control_seeds()
    plot_checkpoints()
    plot_checkpoint_correlations()
    plot_ablation_specific_evaluations(annotate=True)
    plot_ablation_grid(annotate=True)
    plot_ablation_specific_evaluations(annotate=False)
    plot_ablation_grid(annotate=False)
    print("\nDone!")