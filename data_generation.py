import numpy as np

taper = np.ones((90, 1), dtype=np.float32)
taper[:5, 0] = np.linspace(0, 1, 5)   
taper[-5:, 0] = np.linspace(1, 0, 5)  

def generate_event(num_spikes, encoder, decoder, X_train_raw, X_train_norm, noise_matrix):
    max_val = np.max(np.abs(X_train_raw))
    final_clean_event = np.zeros((90, 32), dtype=np.float32)
    shifts = []

    if num_spikes == 1:
        idx = np.random.choice(len(X_train_raw))
        spike = X_train_raw[idx]
        shift = np.random.randint(0, 60)

        write_len = min(spike.shape[0], 90 - shift)
        final_clean_event[shift : shift + write_len] = spike[:write_len]
        shifts.append(shift)

    else:
        for i in range(num_spikes):
            idx = np.random.choice(len(X_train_norm))
            z, _, _ = encoder.predict(X_train_norm[idx:idx+1], verbose=0)
            z_novel = z + np.random.normal(0, 0.05, size=z.shape)
            spike = decoder.predict(z_novel, verbose=0)[0] * max_val

            shift = 0 if i == 0 else np.random.randint(1, 5)

            shifted_spike = np.zeros_like(spike)
            if shift > 0:
                shifted_spike[shift:] = spike[:-shift]
            else:
                shifted_spike = spike

            final_clean_event += shifted_spike
            shifts.append(shift)

    noise_start = np.random.randint(0, len(noise_matrix) - 90)
    noise_window = noise_matrix[noise_start : noise_start + 90]
    final_hybrid_event = final_clean_event + noise_window

    return final_hybrid_event, shifts

def build_dataset(num_samples, p_distribution, X_train_raw, X_train_reconstructed, drift_noise, footprints_norm):
    print(f"Generating {num_samples} VAE-Augmented adversarial samples...")
    spike_counts = [0, 1, 2, 3, 4, 5]
    n_spikes_per_sample = np.random.choice(spike_counts, size=num_samples, p=p_distribution)
    max_val = np.percentile(np.abs(X_train_raw), 99.9)

    X_batch = np.zeros((num_samples, 90, 32), dtype=np.float32)
    y_batch = np.zeros((num_samples, 90, 1), dtype=np.float32)
    noise_starts = np.random.randint(0, len(drift_noise) - 90, size=num_samples)
    x_axis = np.arange(90)

    for i in range(num_samples):
        n_spikes = n_spikes_per_sample[i]
        final_physical_event = np.zeros((90, 32), dtype=np.float32)

        if n_spikes > 0:
            anchor_shift = np.random.randint(10, 45)
            anchor_idx = np.random.randint(0, len(X_train_raw))
            selected_indices = [anchor_idx]

            if n_spikes > 1:
                anchor_vector = footprints_norm[anchor_idx]
                similarities = np.dot(footprints_norm, anchor_vector)
                valid_indices = np.where((similarities > 0.85) & (np.arange(len(similarities)) != anchor_idx))[0]
                if len(valid_indices) < n_spikes - 1:
                    valid_indices = np.where((similarities > 0.50) & (np.arange(len(similarities)) != anchor_idx))[0]
                correlated_picks = np.random.choice(valid_indices, size=n_spikes - 1, replace=True)
                selected_indices.extend(correlated_picks)

            for s, idx in enumerate(selected_indices):
                raw_spike_unaltered = X_train_raw[idx]
                reconstructed_spike_tapered = X_train_reconstructed[idx] * taper

                shift = anchor_shift if s == 0 else anchor_shift + np.random.randint(0, 20)

                if shift > 0:
                    write_len = min(reconstructed_spike_tapered.shape[0], 90 - shift)
                    final_physical_event[shift : shift + write_len] += reconstructed_spike_tapered[:write_len]
                else:
                    final_physical_event += reconstructed_spike_tapered
                best_channel = np.argmax(np.ptp(reconstructed_spike_tapered, axis=0))
                
                physical_peak_idx = np.argmin(reconstructed_spike_tapered[:, best_channel])
                absolute_peak_pos = shift + physical_peak_idx

                sigma = 1.0
                if 0 <= absolute_peak_pos < 90:
                    gaussian = np.exp(-((x_axis - absolute_peak_pos)**2) / (2 * sigma**2))
                    y_batch[i, :, 0] += gaussian

            y_batch[i] = np.clip(y_batch[i], 0.0, 1.0)

        noise_window = drift_noise[noise_starts[i] : noise_starts[i] + 90]
        noise_window = noise_window - np.mean(noise_window, axis=0)
        final_hybrid_event = final_physical_event + (noise_window * 1.0)

        X_batch[i] = np.clip(final_hybrid_event / (max_val + 1e-8), -10.0, 10.0)

    return X_batch, y_batch

def build_strict_raw_test_set(num_samples, p_distribution, raw_spike_pool, drift_noise, footprints_norm_test):
    print(f"Generating {num_samples} strict raw samples with physical noise...")
    spike_counts = [0, 1, 2, 3, 4, 5]
    n_spikes_per_sample = np.random.choice(spike_counts, size=num_samples, p=p_distribution)
    max_val = np.percentile(np.abs(raw_spike_pool), 99.9)

    X_batch = np.zeros((num_samples, 90, 32), dtype=np.float32)
    y_batch = np.zeros((num_samples, 90, 1), dtype=np.float32)
    noise_starts = np.random.randint(0, len(drift_noise) - 90, size=num_samples)
    x_axis = np.arange(90)

    for i in range(num_samples):
        n_spikes = n_spikes_per_sample[i]
        final_physical_event = np.zeros((90, 32), dtype=np.float32)

        if n_spikes > 0:
            anchor_shift = np.random.randint(10, 45)
            anchor_idx = np.random.randint(0, len(raw_spike_pool))
            selected_indices = [anchor_idx]

            if n_spikes > 1:
                anchor_vector = footprints_norm_test[anchor_idx]
                similarities = np.dot(footprints_norm_test, anchor_vector)
                valid_indices = np.where((similarities > 0.85) & (np.arange(len(similarities)) != anchor_idx))[0]
                if len(valid_indices) < n_spikes - 1:
                    valid_indices = np.where((similarities > 0.50) & (np.arange(len(similarities)) != anchor_idx))[0]
                correlated_picks = np.random.choice(valid_indices, size=n_spikes - 1, replace=True)
                selected_indices.extend(correlated_picks)

            for s, idx in enumerate(selected_indices):
                raw_spike_unaltered = raw_spike_pool[idx]
                raw_spike_tapered = raw_spike_unaltered * taper

                shift = anchor_shift if s == 0 else anchor_shift + np.random.randint(0, 20)

                if shift > 0:
                    write_len = min(raw_spike_tapered.shape[0], 90 - shift)
                    final_physical_event[shift : shift + write_len] += raw_spike_tapered[:write_len]
                else:
                    final_physical_event += raw_spike_tapered

                best_channel = np.argmax(np.ptp(raw_spike_unaltered, axis=0))
                physical_peak_idx = np.argmin(raw_spike_unaltered[:, best_channel])
                absolute_peak_pos = shift + physical_peak_idx

                sigma = 1.0
                if 0 <= absolute_peak_pos < 90:
                    gaussian = np.exp(-((x_axis - absolute_peak_pos)**2) / (2 * sigma**2))
                    y_batch[i, :, 0] += gaussian

            y_batch[i] = np.clip(y_batch[i], 0.0, 1.0)

        noise_window = drift_noise[noise_starts[i] : noise_starts[i] + 90]
        noise_window = noise_window - np.mean(noise_window, axis=0)
        final_hybrid_event = final_physical_event + (noise_window * 1.0)

        X_batch[i] = np.clip(final_hybrid_event / (max_val + 1e-8), -10.0, 10.0)

    return X_batch, y_batch