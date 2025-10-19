import torch
import dpp
from dpp.data import Batch
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
import argparse
import os
import scipy
import scipy.integrate
import warnings
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import bds

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
def load_models(config):
    """
    Load the target and draft models.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        target_model: The larger target model
        draft_model: The smaller draft model
    """
    target_path = config['sampling']['model_paths']['target']
    draft_path = config['sampling']['model_paths']['draft']

    # Load target model
    target_model = dpp.models.LogNormMixTransformer(
        num_marks=config.get('num_marks', 2),
        encoder_type=config['model_type'],
        mean_log_inter_time=config.get('mean_log_inter_time', 0.0),
        std_log_inter_time=config.get('std_log_inter_time', 1.0),
        context_size=config['target_model']['context_size'],
        num_mix_components=config['target_model']['num_mix_components'],
        nhead=config['target_model']['transformer']['nhead'],
        num_layers=config['target_model']['transformer']['num_layers'],
        dropout=config['target_model']['transformer']['dropout']
    )
    target_model.load_state_dict(torch.load(target_path, map_location='cpu', weights_only=True))
    print('Target model loaded')
    
    # Load draft model
    draft_model = dpp.models.LogNormMixTransformer(
        num_marks=config.get('num_marks', 2),
        encoder_type=config['model_type'],
        mean_log_inter_time=config.get('mean_log_inter_time', 0.0),
        std_log_inter_time=config.get('std_log_inter_time', 1.0),
        context_size=config['draft_model']['context_size'],
        num_mix_components=config['draft_model']['num_mix_components'],
        nhead=config['draft_model']['transformer']['nhead'],
        num_layers=config['draft_model']['transformer']['num_layers'],
        dropout=config['draft_model']['transformer']['dropout']
    )
    draft_model.load_state_dict(torch.load(draft_path, map_location='cpu', weights_only=True))
    print('Draft model loaded')
    
    return target_model, draft_model

def plot_sampling_results(results, save_path=None):
    """
    Plot results from the benchmark comparison.
    
    Args:
        results: Dictionary with benchmark results
        save_path: Path to save the figure (optional)
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot event times
    ax = axs[0, 0]
    batch_idx = 0  # Use the first batch for plotting
    history_len = 1  # We start with one event
    
    # Get times data
    historic_times = results["real"]["times"][batch_idx, history_len:].cpu().numpy()
    standard_times = results["standard"]["times"][batch_idx, history_len:].cpu().numpy()
    spec_times = results["speculative"]["times"][batch_idx, history_len:].cpu().numpy()
    
    # Find the upper bound for x-axis to better position the legend
    max_time = max(np.max(historic_times) if historic_times.size > 0 else 0,
                   np.max(standard_times) if standard_times.size > 0 else 0, 
                   np.max(spec_times) if spec_times.size > 0 else 0)
    
    # Plot historic events
    ax.scatter(
        historic_times, 
        np.zeros_like(historic_times) - 0.05, 
        marker='|', 
        s=100, 
        color='orange', 
        label='Real events'
    )
    
    # Plot standard sampling
    ax.scatter(
        standard_times, 
        np.zeros_like(standard_times), 
        marker='|', 
        s=100, 
        color='blue', 
        label='Standard sampling'
    )
    
    # Plot speculative sampling
    ax.scatter(
        spec_times, 
        np.zeros_like(spec_times) + 0.05, 
        marker='|', 
        s=100, 
        color='red', 
        alpha=0.7,
        label='Speculative sampling'
    )
    
    ax.set_title('Event Time Comparison')
    ax.set_xlabel('Time')
    ax.set_yticks([])
    
    # Position the legend in the upper right but ensure it doesn't overlap with events
    # by limiting the x-axis range and placing the legend outside
    ax.set_xlim(0, max_time * 1.05)  # Add 5% margin
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    
    # Plot runtime comparison
    ax = axs[0, 1]
    methods = ['Standard', 'Speculative']
    times = [results["standard"]["duration"], results["speculative"]["duration"]]
    ax.bar(methods, times, color=['blue', 'red'])
    ax.set_title('Runtime Comparison')
    ax.set_ylabel('Runtime (seconds)')
    
    for i, v in enumerate(times):
        ax.text(i, v + 0.1, f"{v:.2f}s", ha='center')
    
    ax.text(0.5, 0.9, f"Speedup: {results['speedup']:.2f}x", 
            transform=ax.transAxes, ha='center', fontsize=14, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot events generated
    ax = axs[1, 0]
    events = [results["standard"]["num_events"], results["speculative"]["num_events"]]
    ax.bar(methods, events, color=['blue', 'red'])
    ax.set_title('Events Generated')
    ax.set_ylabel('Number of events')
    
    for i, v in enumerate(events):
        ax.text(i, v + 0.1, str(v), ha='center')
    
    # Plot speculative decoding metrics
    ax = axs[1, 1]
    metrics = results["speculative"]["metrics"]
    metric_names = ['Acceptance Rate', 'Events per Iteration']
    metric_values = [metrics["acceptance_rate"], metrics["events_per_iteration"]]
    ax.bar(metric_names, metric_values, color=['green', 'purple'])
    ax.set_title('Speculative Decoding Metrics')
    ax.set_ylim(0, max(2.0, max(metric_values) * 1.2))
    
    for i, v in enumerate(metric_values):
        ax.text(i, v + 0.05, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_ll_comparison(results, save_path=None):
    """
    Create a bar plot comparing the negative log-likelihood (NLL) of the standard and 
    speculative sampling methods.
    
    Args:
        results: Dictionary with results from benchmark_comparison
        save_path: Path to save the plot
        
    Returns:
        The matplotlib figure
    """
    
    # Extract NLL values
    std_ll = results["ll"]["standard"]
    spec_ll = results["ll"]["speculative"]
    real_ll = results["ll"]["real"] if "real" in results["ll"] else None
    print(f'Standard LL: {std_ll}, Speculative LL: {spec_ll}, Real LL: {real_ll}')
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    if real_ll is not None:
        methods = ['Real', 'Standard', 'Speculative']
        values = [real_ll, std_ll, spec_ll]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
    else:
        methods = ['Standard', 'Speculative']
        values = [std_ll, spec_ll]
        colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(methods, values, color=colors, width=0.6)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{values[i]:.4f}',
                ha='center', va='bottom', fontsize=12)
    
    ax.set_ylabel('Normalized Log-Likelihood', fontsize=14)
    ax.set_title('Standard vs Speculative Sampling', fontsize=16)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust y-axis to start slightly below the minimum value for better visualization
    
    ax.set_ylim(-5, 10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    ax.legend(bars, methods, loc='upper right', fontsize=12)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"LL comparison plot saved to {save_path}")
    
    return fig

def aggregate_ll_over_synth_dataset(dl):
    total_ll = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in dl:
            if batch.nlls is not None:
                batch_avg_nlls = batch.nlls / batch.t_ends
                total_ll += -batch_avg_nlls.sum().item()
                total_count += batch_avg_nlls.numel()
            else:
                print("Warning: Batch does not contain nlls.")
                break
    if total_count == 0: return None
    else: return total_ll / total_count

def add_synth_dataset(dl, results, config):
    batches = list(dl)
    np.random.seed(config['sampling']['gamma'])
    idx1 = 0
    idx2, idx3 = np.random.random_integers(0,len(dl),size=2)

    results["real"]["times"] = batches[idx1].event_times
    results["real"]["marks"] = batches[idx1].marks
    results["real"]["duration"] = batches[idx1].t_ends
    results["real"]["metrics"] = None
    results["real"]["num_events"] = batches[idx1].event_times.shape[1]
    real_ll = aggregate_ll_over_synth_dataset(dl)
    results["ll"]["real"] = real_ll
    
    # results["standard"]["times"] = batches[idx2].event_times
    # results["standard"]["marks"] = batches[idx2].marks
    
    # results["speculative"]["times"] = batches[idx3].event_times
    # results["speculative"]["marks"] = batches[idx3].marks


def plot_temporal_goodness_of_fit(results, type=None, save_path=None):
    """
    Plot goodness-of-fit diagnostics for point process data using time rescaling theorem.
    
    Args:
        results: Dictionary with benchmark results
        type: Type of point process ('hawkes' or 'inhomo-poisson')
        save_path: Path to save the figures (optional)
    """
    # Define ground truth intensity functions
    gt_intensity = {
        "hawkes": [2.5, 1.0, 2.0, 100],
        "inhomo-poisson": lambda t: 5*(1+np.sin(2*np.pi/100*t)),
        "self-correct": [0.3, 0.3, 1000],
        "multi-hawkes": [[0.5, 0.5], [[1, 0.5],[0.1, 1]], [[2, 2],[2, 2]]]
    }
    
    # Extract and preprocess times
    spec_times = results["speculative"]["times"].cpu().numpy()
    std_times = results["standard"]["times"].cpu().numpy()
    real_times = results["real"]["times"].cpu().numpy()
    
    if type == 'multi-hawkes':
        # Extract marks
        spec_marks = results["speculative"]["marks"].cpu().numpy()
        std_marks = results["standard"]["marks"].cpu().numpy()
        real_marks = results["real"]["marks"].cpu().numpy()
        
        # Filter to get only the first dimension (mark = 0)
        spec_times_dim1 = spec_times[spec_marks == 0]
        std_times_dim1 = std_times[std_marks == 0]
        real_times_dim1 = real_times[real_marks == 0]
        
        # Remove padding zeros (assuming 0 is used for padding)
        spec_times = spec_times_dim1[spec_times_dim1 > 0]
        std_times = std_times_dim1[std_times_dim1 > 0]
        real_times = real_times_dim1[real_times_dim1 > 0]
    else:
        # Remove padding zeros (assuming 0 is used for padding)
        spec_times = spec_times[spec_times > 0]
        std_times = std_times[std_times > 0]
        real_times = real_times[real_times > 0]
    
    # Sort the times (ensuring they're in chronological order)
    spec_times = np.sort(spec_times)
    std_times = np.sort(std_times)
    real_times = np.sort(real_times)

    # Check if type is valid
    if type not in gt_intensity:
        print(f"Error: type '{type}' is not valid. Must be one of {list(gt_intensity.keys())}")
        return None
    
    # Compute integrated intensities based on process type
    if type == 'hawkes':
        params = gt_intensity[type]

        def hawkes_intensity(params, t, history):
            mu = params[0]
            alpha = params[1]
            beta = params[2]
            
            intensity = mu
            for t_i in history:
                if t_i < t:
                    intensity += alpha * np.exp(-beta * (t - t_i))
            return intensity
        
        # Function to compute the integrated conditional intensity (Λ(t))
        def compute_integrated_intensity(params, times):
            mu = params[0]
            alpha = params[1]
            beta = params[2]

            z = np.zeros_like(times)
            for i, t in enumerate(times):
                # History includes all events before current time
                history = times[:i]
                
                # For the first event, we can compute analytically
                if i == 0:
                    z[i] = mu * t
                else:
                    # For subsequent events, we need to add the contribution of previous events
                    prev_z = z[i-1]
                    delta_t = t - times[i-1]
                    
                    # Compute intensity at previous event
                    prev_intensity = hawkes_intensity(params, times[i-1], history)
                    
                    # Add contribution of base intensity
                    contrib_base = mu * delta_t
                    
                    # Add contribution of kernels from previous events
                    contrib_history = 0
                    for t_j in history:
                        # Integral of α*exp(-β(t-t_j)) from times[i-1] to t
                        contrib_history += (alpha/beta) * (np.exp(-beta * (times[i-1] - t_j)) - 
                                                        np.exp(-beta * (t - t_j)))
                    
                    # Total integrated intensity
                    z[i] = prev_z + contrib_base + contrib_history
            
            return z
        
        # Compute transformed inter-event times using time rescaling theorem
        spec_z = compute_integrated_intensity(params, spec_times)
        std_z = compute_integrated_intensity(params, std_times)
        real_z = compute_integrated_intensity(params, real_times)
        spec_transformed = np.diff(spec_z)
        std_transformed = np.diff(std_z)
        real_transformed = np.diff(real_z)

    elif type == 'inhomo-poisson':
        intensity_func = gt_intensity[type]
        
        # Function to compute the integrated intensity (Λ(t)) for inhomogeneous Poisson
        def compute_integrated_intensity(times, intensity_func):
            # For inhomogeneous Poisson, we need to compute ∫λ(s)ds from 0 to t
            # This is done using numerical integration
            z = np.zeros_like(times)
            for i, t in enumerate(times):
                # Numerical integration of intensity function from 0 to t
                z[i], _ = scipy.integrate.quad(intensity_func, 0, t)
            return z
        
        # Compute transformed inter-event times using time rescaling theorem
        spec_z = compute_integrated_intensity(spec_times, intensity_func)
        std_z = compute_integrated_intensity(std_times, intensity_func)
        real_z = compute_integrated_intensity(real_times, intensity_func)
        spec_transformed = np.diff(spec_z)
        std_transformed = np.diff(std_z)
        real_transformed = np.diff(real_z)
    
    elif type == 'self-correct':
        params = gt_intensity[type]

        # For a self-correcting process, the intensity is:
        # λ(t) = exp(μt - α*N(t))
        
        def compute_integrated_intensity_selfcorrect(times, params):
            mu = params[0]
            alpha = params[1]
            
            if len(times) == 0:
                return np.array([])
            
            # Initialize the transformed array to include all intervals
            # including the first one from 0 to t₁
            transformed = np.zeros(len(times))
            
            # Calculate the first interval (from 0 to t₁)
            # N(0) = 0, so the computation is simply:
            # ∫₀^{t₁} exp(μs)ds = (exp(μ*t₁) - 1)/μ
            if len(times) > 0:
                transformed[0] = (np.exp(mu * times[0]) - 1) / mu
            
            # Calculate the remaining intervals
            for i in range(1, len(times)):
                t_prev = times[i-1]
                t_curr = times[i]
                
                # N(t) at t_prev is i
                N_t_prev = i
                
                # Compute integrated intensity from t_prev to t_curr
                transformed[i] = np.exp(-alpha * N_t_prev) * (np.exp(mu * t_curr) - np.exp(mu * t_prev)) / mu
            
            return transformed
    
        # Compute transformed inter-event times using time rescaling theorem
        spec_transformed = compute_integrated_intensity_selfcorrect(spec_times, params)
        std_transformed = compute_integrated_intensity_selfcorrect(std_times, params)
        real_transformed = compute_integrated_intensity_selfcorrect(real_times, params)     

    elif type == 'multi-hawkes':
        params = gt_intensity[type]
        
        # Extract all event times for all dimensions
        spec_all_times = results["speculative"]["times"].cpu().numpy()
        std_all_times = results["standard"]["times"].cpu().numpy()
        real_all_times = results["real"]["times"].cpu().numpy()
        
        spec_all_marks = results["speculative"]["marks"].cpu().numpy()
        std_all_marks = results["standard"]["marks"].cpu().numpy()
        real_all_marks = results["real"]["marks"].cpu().numpy()
        
        # Remove padding zeros from all times arrays
        spec_valid_indices = spec_all_times > 0
        std_valid_indices = std_all_times > 0
        real_valid_indices = real_all_times > 0
        
        spec_all_times = spec_all_times[spec_valid_indices]
        spec_all_marks = spec_all_marks[spec_valid_indices]
        std_all_times = std_all_times[std_valid_indices]
        std_all_marks = std_all_marks[std_valid_indices]
        real_all_times = real_all_times[real_valid_indices]
        real_all_marks = real_all_marks[real_valid_indices]
        
        # Sort all times and marks together
        spec_sorted_indices = np.argsort(spec_all_times)
        spec_all_times = spec_all_times[spec_sorted_indices]
        spec_all_marks = spec_all_marks[spec_sorted_indices]
        
        std_sorted_indices = np.argsort(std_all_times)
        std_all_times = std_all_times[std_sorted_indices]
        std_all_marks = std_all_marks[std_sorted_indices]
        
        real_sorted_indices = np.argsort(real_all_times)
        real_all_times = real_all_times[real_sorted_indices]
        real_all_marks = real_all_marks[real_sorted_indices]
        
        # Function to compute the integrated conditional intensity for first dimension
        def compute_cumulative_intensity_multihawkes_dim1(evaluation_times, all_times, all_marks, params):
            mu_1 = params[0][0]
            alpha = params[1]
            beta = params[2]
            num_dims = len(params[0])
            
            cumulative_intensity = np.zeros_like(evaluation_times)
            
            for i, t in enumerate(evaluation_times):
                base_contrib = mu_1 * t
                history_mask = all_times < t
                history_times = all_times[history_mask]
                history_marks = all_marks[history_mask]
                
                history_contrib = 0.0
                for t_j, mark_j in zip(history_times, history_marks):
                    a_1n = alpha[0][mark_j]
                    b_1n = beta[0][mark_j]
                    
                    # ∫_{t_j}^t a_1n * exp(-b_1n * (s - t_j)) ds
                    contrib = (a_1n / b_1n) * (1 - np.exp(-b_1n * (t - t_j)))
                    history_contrib += contrib
                
                cumulative_intensity[i] = base_contrib + history_contrib
            
            return cumulative_intensity
        
        # Compute transformed inter-event times using time rescaling theorem
        spec_z = compute_cumulative_intensity_multihawkes_dim1(spec_times, spec_all_times, spec_all_marks, params)
        std_z = compute_cumulative_intensity_multihawkes_dim1(std_times, std_all_times, std_all_marks, params)
        real_z = compute_cumulative_intensity_multihawkes_dim1(real_times, real_all_times, real_all_marks, params)
        
        spec_transformed = np.diff(spec_z, prepend=0)
        std_transformed = np.diff(std_z, prepend=0)
        real_transformed = np.diff(real_z, prepend=0)
    
    # Create figure with 3 subplots (CDF, KS-plot, QQ-plot)
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 1. Plot empirical CDF vs theoretical CDF
    axes[0].set_title("Empirical vs Exp(1) CDF", fontsize=24)
    
    # Sort the transformed intervals
    spec_sorted = np.sort(spec_transformed)
    std_sorted = np.sort(std_transformed)
    real_sorted = np.sort(real_transformed)
    
    # Create empirical CDFs
    n_spec = len(spec_sorted)
    n_std = len(std_sorted)
    n_real = len(real_sorted)

    spec_ecdf_y = np.arange(1, n_spec + 1) / n_spec
    std_ecdf_y = np.arange(1, n_std + 1) / n_std
    real_ecdf_y = np.arange(1, n_real + 1) / n_real
    
    # Theoretical CDF (exponential with rate 1)
    theoretical_cdf = lambda x: 1 - np.exp(-x)
    
    # Plot empirical and theoretical CDFs
    axes[0].step(spec_sorted, spec_ecdf_y, where='post', label='Speculative', color='blue')
    axes[0].step(std_sorted, std_ecdf_y, where='post', label='Standard', color='green')
    axes[0].step(real_sorted, real_ecdf_y, where='post', label='Real', color='orange')
    
    # Add theoretical CDF line
    x_range = np.linspace(0, max(spec_sorted.max(), std_sorted.max(), real_sorted.max()), 1000)
    axes[0].plot(x_range, theoretical_cdf(x_range), 'r--', label='Exp(1)')
    
    axes[0].set_xlabel('Transformed Inter-event Time', fontsize=24)
    axes[0].set_ylabel('Cumulative Probability', fontsize=24)
    axes[0].legend(fontsize=18)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Create KS plot
    axes[1].set_title("KS Plot", fontsize=24)
    
    # Compute theoretical quantiles based on empirical CDF values
    spec_theoretical_cdf = 1 - np.exp(-spec_sorted)
    std_theoretical_cdf = 1 - np.exp(-std_sorted)
    real_theoretical_cdf = 1 - np.exp(-real_sorted)
    
    # Plot KS plot (empirical CDF vs theoretical CDF)
    axes[1].scatter(spec_theoretical_cdf, spec_ecdf_y, label='Speculative', alpha=0.7, color='blue', s=8)
    axes[1].scatter(std_theoretical_cdf, std_ecdf_y, label='Standard', alpha=0.7, color='green', s=8)
    axes[1].scatter(real_theoretical_cdf, real_ecdf_y, label='Real', alpha=0.7, color='orange', s=8)
    
    # Add 45-degree line
    max_val = max(
        spec_theoretical_cdf.max(), 
        std_theoretical_cdf.max(),
        real_theoretical_cdf.max(),
        spec_ecdf_y.max(),
        std_ecdf_y.max(),
        real_ecdf_y.max()
    )
    axes[1].plot([0, max_val], [0, max_val], 'r--', label='45° Line')
    
    # Add confidence bounds (95%)
    conf_width = 1.36 / np.sqrt(min(n_spec, n_std, n_real))  # 95% confidence bounds for KS test
    upper_bound = lambda x: np.minimum(x + conf_width, 1)  # Ensure bounds don't exceed 1
    lower_bound = lambda x: np.maximum(x - conf_width, 0)  # Ensure bounds don't go below 0
    
    x_range = np.linspace(0, max_val, 1000)
    axes[1].plot(x_range, upper_bound(x_range), 'k:', label='95% CI')
    axes[1].plot(x_range, lower_bound(x_range), 'k:')
    
    axes[1].set_xlabel('Theoretical Cumulative Density', fontsize=24)
    axes[1].set_ylabel('Empirical Cumulative Density', fontsize=24)
    axes[1].legend(fontsize=18)
    axes[1].grid(True, alpha=0.3)
    
    # # 3. Create QQ plot
    # axes[2].set_title("QQ Plot", fontsize=24)
    
    # # Compute theoretical quantiles for uniform percentiles
    # percentiles = np.linspace(0.01, 0.99, 99)
    # theoretical_quantiles = -np.log(1 - percentiles)
    
    # # Compute empirical quantiles
    # spec_empirical_quantiles = np.percentile(spec_transformed, 100 * percentiles)
    # std_empirical_quantiles = np.percentile(std_transformed, 100 * percentiles)
    # real_empirical_quantiles = np.percentile(real_transformed, 100 * percentiles)
    
    # # Plot QQ plot
    # axes[2].scatter(theoretical_quantiles, spec_empirical_quantiles, label='Speculative', alpha=0.7, color='blue', s=8)
    # axes[2].scatter(theoretical_quantiles, std_empirical_quantiles, label='Standard', alpha=0.7, color='green', s=8)
    # axes[2].scatter(theoretical_quantiles, real_empirical_quantiles, label='Real', alpha=0.7, color='orange', s=8)
    
    # # Add 45-degree line
    # max_qq = max(
    #     theoretical_quantiles.max(),
    #     spec_empirical_quantiles.max(),
    #     std_empirical_quantiles.max(),
    #     real_empirical_quantiles.max()
    # )
    # axes[2].plot([0, max_qq], [0, max_qq], 'r--', label='45° Line')
    
    # axes[2].set_xlabel('Theoretical Quantiles', fontsize=24)
    # axes[2].set_ylabel('Empirical Quantiles', fontsize=24)
    # axes[2].legend(fontsize=18)
    # axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Goodness of fit plots saved to {save_path}")
    
    plt.show()
    
    # Calculate KS statistics
    # Maximum deviation from the 45-degree line
    ks_stat_spec = np.max(np.abs(spec_ecdf_y - spec_theoretical_cdf))
    ks_stat_std = np.max(np.abs(std_ecdf_y - std_theoretical_cdf))
    ks_stat_real = np.max(np.abs(real_ecdf_y - real_theoretical_cdf))
    
    print(f"KS statistic for speculative sampling: {ks_stat_spec:.4f}")
    print(f"KS statistic for standard sampling: {ks_stat_std:.4f}")
    print(f"KS statistic for real data: {ks_stat_real:.4f}")
    
    return {
        "ks_stat_spec": ks_stat_spec,
        "ks_stat_std": ks_stat_std,
        "ks_stat_real": ks_stat_real,
        "spec_transformed": spec_transformed,
        "std_transformed": std_transformed,
        "real_transformed": real_transformed
    }
