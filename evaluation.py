import numpy as np
import tensorflow as tf
from scipy.signal import find_peaks

def get_robust_callbacks():
    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", 
        mode="min",        
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",  
        mode="min",          
        factor=0.5,
        patience=1,
        min_lr=1e-6,
        verbose=1
    )

    return [early_stopper, lr_reducer]

def evaluate_predictions(y_true_batch, y_pred_batch, threshold=0.2, tolerance=3, peak_distance=3):
    TP = 0
    FP = 0
    FN = 0
    total_true_spikes = 0

    for i in range(len(y_true_batch)):
        true_indices, _ = find_peaks(y_true_batch[i, :, 0], height=0.5)
        total_true_spikes += len(true_indices)

        pred_probs = y_pred_batch[i, :, 0]
        pred_indices, _ = find_peaks(pred_probs, height=threshold, distance=peak_distance)

        matched_true = set()

        for p_idx in pred_indices:
            matches = [t_idx for t_idx in true_indices if abs(t_idx - p_idx) <= tolerance]
            valid_matches = [m for m in matches if m not in matched_true]

            if valid_matches:
                TP += 1
                closest_match = min(valid_matches, key=lambda x: abs(x - p_idx))
                matched_true.add(closest_match)
            else:
                FP += 1

        FN += (len(true_indices) - len(matched_true))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return TP, FP, FN, precision, recall, f1_score, total_true_spikes

def evaluate_stratified(model, X_test, y_test, model_name, threshold=0.3):
    print(f"\n==================================================")
    print(f" EVALUATING: {model_name}")
    print(f"==================================================")

    y_pred = model.predict(X_test, batch_size=64, verbose=0)

    tier_masks = {
        "Tier 0: Pure Noise (0 Spikes)": [],
        "Tier 1: Isolated Singles (1 Spike)": [],
        "Tier 2: Simple Overlaps (2 Spikes)": [],
        "Tier 3: Complex Bursts (3-5 Spikes)": []
    }

    for i in range(len(y_test)):
        true_peaks, _ = find_peaks(y_test[i, :, 0], height=0.5)
        count = len(true_peaks)

        if count == 0:
            tier_masks["Tier 0: Pure Noise (0 Spikes)"].append(i)
        elif count == 1:
            tier_masks["Tier 1: Isolated Singles (1 Spike)"].append(i)
        elif count == 2:
            tier_masks["Tier 2: Simple Overlaps (2 Spikes)"].append(i)
        elif count >= 3:
            tier_masks["Tier 3: Complex Bursts (3-5 Spikes)"].append(i)

    f1_scores = {}

    for tier_name, indices in tier_masks.items():
        print(f"\n>>> {tier_name} <<<")

        if not indices:
            print("No samples found in this category.")
            f1_scores[tier_name] = 0.0
            continue

        y_test_tier = y_test[indices]
        y_pred_tier = y_pred[indices]

        tp, fp, fn, precision, recall, f1, total_spikes = evaluate_predictions(
            y_test_tier,
            y_pred_tier,
            threshold=threshold,
            tolerance=3,
            peak_distance=5
        )

        f1_scores[tier_name] = f1

        print(f"Total True Spikes: {total_spikes}")
        print(f"True Positives:    {tp}")
        print(f"False Positives:   {fp}")
        print(f"False Negatives:   {fn}")
        print("-" * 30)
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

    return f1_scores

def get_indices_by_complexity(y_true_data, target_spike_count):
    matching_indices = []

    for i in range(len(y_true_data)):
        true_peaks, _ = find_peaks(y_true_data[i, :, 0], height=0.5)

        if len(true_peaks) == target_spike_count:
            matching_indices.append(i)

    return matching_indices
def find_optimal_threshold(model, X_test, y_test, thresholds, peak_distance=5):
    """
    Sweeps through a list of thresholds and returns the one that maximizes 
    the global F1-Score for the given model.
    """
    # Generate predictions exactly once to save compute time
    y_pred = model.predict(X_test, batch_size=64, verbose=0)
    
    best_thresh = thresholds[0]
    best_f1 = -1.0
    
    for thresh in thresholds:
        # Call the core math engine silently
        tp, fp, fn, precision, recall, f1, total_spikes = evaluate_predictions(
            y_test, 
            y_pred, 
            threshold=thresh, 
            tolerance=3, 
            peak_distance=peak_distance
        )
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    return best_thresh