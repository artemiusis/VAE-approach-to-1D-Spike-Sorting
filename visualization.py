import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

def plot_money_shot(X_window, y_true_window, pred_probs_A, pred_probs_B, threshold_A=0.3, threshold_B=0.6, distance=6):
    best_channel = np.argmax(np.max(np.abs(X_window), axis=0))
    waveform = X_window[:, best_channel]

    true_peaks, _ = find_peaks(y_true_window[:, 0], height=0.5)
    preds_A, _ = find_peaks(pred_probs_A[:, 0], height=threshold_A, distance=distance)
    preds_B, _ = find_peaks(pred_probs_B[:, 0], height=threshold_B, distance=distance)

    plt.figure(figsize=(12, 6), dpi=300)
    plt.plot(waveform, color='#2c3e50', linewidth=2.5, label='Raw Extracellular Waveform')

    for tp in true_peaks:
        plt.axvspan(tp - 1, tp + 1, color='#2ecc71', alpha=0.2, label='Ground Truth (Real Spikes)' if tp == true_peaks[0] else "")

    if len(preds_A) > 0:
        plt.scatter(preds_A, waveform[preds_A], color='#e74c3c', marker='x', s=150, linewidths=3, zorder=5, label='Model A (Baseline)')

    if len(preds_B) > 0:
        plt.scatter(preds_B, waveform[preds_B] + 0.1, facecolors='none', edgecolors='#3498db', marker='o', s=200, linewidths=3, zorder=5, label='Model B (VAE Augmented)')

    plt.title("Resolving Complex Biological Overlaps: Baseline vs. VAE", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Time (Samples)", fontsize=12, fontweight='bold')
    plt.ylabel("Normalized Amplitude", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 90)
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=11, framealpha=0.9)
    plt.tight_layout()
    plt.show()

def plot_performance_delta(results_A, results_B, results_C=None):
    tiers = ["Tier 0\n(Pure Noise)", "Tier 1\n(Singles)", "Tier 2\n(Overlaps)", "Tier 3\n(Bursts)"]

    f1_A = [results_A.get(k, 0) for k in ["Tier 0: Pure Noise (0 Spikes)", "Tier 1: Isolated Singles (1 Spike)", "Tier 2: Simple Overlaps (2 Spikes)", "Tier 3: Complex Bursts (3-5 Spikes)"]]
    f1_B = [results_B.get(k, 0) for k in ["Tier 0: Pure Noise (0 Spikes)", "Tier 1: Isolated Singles (1 Spike)", "Tier 2: Simple Overlaps (2 Spikes)", "Tier 3: Complex Bursts (3-5 Spikes)"]]

    if results_C:
        f1_C = [results_C.get(k, 0) for k in ["Tier 0: Pure Noise (0 Spikes)", "Tier 1: Isolated Singles (1 Spike)", "Tier 2: Simple Overlaps (2 Spikes)", "Tier 3: Complex Bursts (3-5 Spikes)"]]

    x = np.arange(len(tiers))
    width = 0.25 if results_C else 0.35

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    if results_C:
        rects1 = ax.bar(x - width, f1_A, width, label='Model A (CNN Baseline)', color='#e74c3c', edgecolor='black')
        rects2 = ax.bar(x, f1_B, width, label='Model B (VAE Augmented)', color='#3498db', edgecolor='black')
        rects3 = ax.bar(x + width, f1_C, width, label='Model C (Linear Mixed)', color='#95a5a6', edgecolor='black')
    else:
        rects1 = ax.bar(x - width/2, f1_A, width, label='Model A (CNN Baseline)', color='#e74c3c', edgecolor='black')
        rects2 = ax.bar(x + width/2, f1_B, width, label='Model B (VAE Augmented)', color='#3498db', edgecolor='black')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0.01:
                ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    if results_C:
        autolabel(rects3)

    ax.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
    ax.set_title('Resolving Extracellular Collisions: Architecture Comparison', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

def plot_vae_artifact(raw_window, encoder, decoder):
    if len(raw_window.shape) == 2:
        raw_window_batch = np.expand_dims(raw_window, axis=0)
    else:
        raw_window_batch = raw_window

    z_mean, _, _ = encoder.predict(raw_window_batch, verbose=0)
    reconstructed_window = decoder.predict(z_mean, verbose=0)[0]

    raw_window = raw_window_batch[0]
    best_channel = np.argmax(np.max(np.abs(raw_window), axis=0))

    raw_waveform = raw_window[:, best_channel]
    reconstructed_waveform = reconstructed_window[:, best_channel]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300, sharey=True)
    fig.suptitle("The VAE Artifact: Non-Linear Denoising & Lossy Compression", fontsize=18, fontweight='bold', y=1.05)

    ax1.plot(raw_waveform, color='#2c3e50', linewidth=2.5)
    ax1.set_title("Input: Raw Extracellular Spike\n(Contains Native Thermal Noise)", fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel("Time (Samples)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Normalized Amplitude", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xlim(0, 90)

    ax2.plot(reconstructed_waveform, color='#3498db', linewidth=2.5)
    ax2.set_title("Output: VAE Reconstruction\n(Sterile, Low-Pass Filtered)", fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel("Time (Samples)", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_xlim(0, 90)

    residual = raw_waveform - reconstructed_waveform
    ax2.fill_between(range(90), 0, residual, color='#e74c3c', alpha=0.15, label="Information Lost (Noise/Jitter)")
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
def plot_statistical_performance(history_A, history_B, history_C):
    """
    Plots a grouped bar chart with error bars representing the standard deviation
    across multiple training runs for three competing models.
    """
    tiers = ["Tier 0\n(Pure Noise)", "Tier 1\n(Singles)", "Tier 2\n(Overlaps)", "Tier 3\n(Bursts)"]
    keys = [
        "Tier 0: Pure Noise (0 Spikes)", 
        "Tier 1: Isolated Singles (1 Spike)", 
        "Tier 2: Simple Overlaps (2 Spikes)", 
        "Tier 3: Complex Bursts (3-5 Spikes)"
    ]
    
    # Calculate means and standard deviations for Model A
    means_A = [np.mean(history_A.get(k, [0])) for k in keys]
    stds_A = [np.std(history_A.get(k, [0])) for k in keys]
    
    # Calculate means and standard deviations for Model B
    means_B = [np.mean(history_B.get(k, [0])) for k in keys]
    stds_B = [np.std(history_B.get(k, [0])) for k in keys]
    
    # Calculate means and standard deviations for Model C
    means_C = [np.mean(history_C.get(k, [0])) for k in keys]
    stds_C = [np.std(history_C.get(k, [0])) for k in keys]

    x = np.arange(len(tiers))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

    # Plot bars with yerr for error bars
    rects1 = ax.bar(x - width, means_A, width, yerr=stds_A, label='Model A (CNN Baseline)', color='#e74c3c', edgecolor='black', capsize=5)
    rects2 = ax.bar(x, means_B, width, yerr=stds_B, label='Model B (VAE Augmented)', color='#3498db', edgecolor='black', capsize=5)
    rects3 = ax.bar(x + width, means_C, width, yerr=stds_C, label='Model C (Linear Mixed)', color='#95a5a6', edgecolor='black', capsize=5)

    def autolabel(rects, means):
        for rect, mean_val in zip(rects, means):
            height = rect.get_height()
            if height > 0.01:
                ax.annotate(f'{mean_val:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 8), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1, means_A)
    autolabel(rects2, means_B)
    autolabel(rects3, means_C)

    ax.set_ylabel('F1-Score (Mean ± Std Dev)', fontsize=14, fontweight='bold')
    ax.set_title('Resolving Extracellular Collisions: Statistical Performance (N=3 Runs)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()