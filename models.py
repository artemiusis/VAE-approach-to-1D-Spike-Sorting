import math
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from evaluation import evaluate_predictions, evaluate_stratified, find_optimal_threshold

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], tf.shape(z_mean)[1]))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mse(data, reconstruction), axis=1))
            kl_loss = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1))
            total_loss = reconstruction_loss + (self.kl_weight * kl_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {"loss": self.total_loss_tracker.result(), "reconstruction": self.reconstruction_loss_tracker.result(), "kl": self.kl_loss_tracker.result()}

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mse(data, reconstruction), axis=1))
        kl_loss = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1))
        total_loss = reconstruction_loss + (self.kl_weight * kl_loss)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {"loss": self.total_loss_tracker.result(), "reconstruction": self.reconstruction_loss_tracker.result(), "kl": self.kl_loss_tracker.result()}

class KLAnnealingCallback(tf.keras.callbacks.Callback):
    def __init__(self, max_weight=0.01, anneal_epochs=20):
        super().__init__()
        self.max_weight = max_weight
        self.anneal_epochs = anneal_epochs

    def on_epoch_begin(self, epoch, logs=None):
        tf.keras.backend.set_value(self.model.kl_weight, min(self.max_weight, (epoch / self.anneal_epochs) * self.max_weight))

import tensorflow as tf
from tensorflow.keras import layers, Model

def build_overlap_classifier(num_samples=90, num_channels=32, learning_rate=0.0005, clipnorm=1.0):
    inputs = layers.Input(shape=(num_samples, num_channels))

    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling1D(pool_size=2, padding="same")(x)

    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)

    x = layers.UpSampling1D(size=2)(x)
    x = layers.Cropping1D(cropping=(0, x.shape[1] - num_samples))(x)

    outputs = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"))(x)

    model = Model(inputs=inputs, outputs=outputs, name="CNN_GRU_Mixer")

    # FIX: Added clipnorm to physically cap destructive gradient updates
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model




def train_and_evaluate_n_runs(X_train_full, y_train_full, sample_weights, X_test, y_test, build_fn, callback_fn, epochs=20, batch_size=64, num_runs=3):
    """
    Trains a model N times, evaluates each run on the test set, 
    calculates average performance across all tiers, and returns the best overall model.
    """
    best_val_f1 = -1.0
    temp_weights_path = "temp_best_weights.weights.h5"

    # Dictionary to track performance across all runs
    tier_history = {
        "Tier 0: Pure Noise (0 Spikes)": [],
        "Tier 1: Isolated Singles (1 Spike)": [],
        "Tier 2: Simple Overlaps (2 Spikes)": [],
        "Tier 3: Complex Bursts (3-5 Spikes)": []
    }

    # Explicitly create a validation split
    X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
        X_train_full, y_train_full, sample_weights, test_size=0.2, random_state=101
    )

    test_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    for run in range(num_runs):
        print(f"\n" + "="*50)
        print(f" EXECUTION INITIALIZATION {run + 1}/{num_runs}")
        print("="*50)
        
        tf.keras.backend.clear_session()
        
        seed = 42 + run
        tf.random.set_seed(seed)
        np.random.seed(seed)

        model = build_fn()
        fresh_callbacks = callback_fn()
        
        model.fit(
            X_train, y_train,
            sample_weight=sw_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val, sw_val),
            callbacks=fresh_callbacks,
            verbose=1
        )

        # --- VALIDATION EVALUATION (For Model Selection) ---
        y_val_pred = model.predict(X_val, batch_size=batch_size, verbose=0)
        current_best_val_f1 = -1.0
        
        for thresh in test_thresholds:
            _, _, _, _, _, f1, _ = evaluate_predictions(
                y_val, y_val_pred, threshold=thresh, tolerance=3, peak_distance=5
            )
            if f1 >= current_best_val_f1:
                current_best_val_f1 = f1

        if current_best_val_f1 > best_val_f1:
            best_val_f1 = current_best_val_f1
            model.save_weights(temp_weights_path)
            print(f">>> New Best Weights Saved! (Val F1: {best_val_f1:.4f}) <<<")

        # --- TEST EVALUATION (For Statistical Tracking) ---
        print(f"\nRunning test evaluation for Run {run + 1}...")
        optimal_test_thresh = find_optimal_threshold(model, X_test, y_test, test_thresholds)
        
        # Suppress the heavy print output for individual runs by modifying evaluate_stratified 
        # or just letting it print. We'll capture the returned dictionary.
        run_results = evaluate_stratified(
            model, X_test, y_test, 
            model_name=f"Run {run + 1} (Thresh: {optimal_test_thresh})", 
            threshold=optimal_test_thresh
        )

        for tier, score in run_results.items():
            if tier in tier_history:
                tier_history[tier].append(score)

    # --- CALCULATE AND PRINT AVERAGES ---
    print("\n" + "="*65)
    print(f" STATISTICAL AVERAGE ACROSS {num_runs} RUNS (F1-SCORE)")
    print("="*65)
    print(f"{'Complexity Tier':<35} | {'Mean F1':<8} | {'Std Dev'}")
    print("-" * 65)
    
    for tier in tier_history.keys():
        scores = tier_history[tier]
        mean_score = np.mean(scores)
        std_dev = np.std(scores)
        short_name = tier.split(":")[0]
        print(f"{short_name:<35} | {mean_score:.4f}  | ±{std_dev:.4f}")
    print("="*65)

    print("\nRestoring absolute best model weights...")
    tf.keras.backend.clear_session()
    final_model = build_fn()
    _ = final_model(X_train[:1]) 
    final_model.load_weights(temp_weights_path)
    
    if os.path.exists(temp_weights_path):
        os.remove(temp_weights_path)
        
    return final_model, tier_history