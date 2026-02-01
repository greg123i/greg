
import os
import time
import json
import math
import statistics
import datetime

class DiagnosticsManager:
    def __init__(self, diagnostics_dir):
        self.diagnostics_dir = diagnostics_dir
        os.makedirs(self.diagnostics_dir, exist_ok=True)
        
        self.history = []
        self.latest_metrics = {
            "timestamp": 0,
            "loss": 0.0,
            "learning_rate": 0.0,
            "grad_norm": 0.0,
            "perplexity": 0.0,
            "tokens_per_sec": 0.0,
            "step": 0,
            "epoch": 0,
            "sample_output": "",
            "anomalies": []
        }
        
        self.last_save_time = time.time()
        self.save_interval_seconds = 10
        self.save_interval_steps = 25
        self.steps_since_save = 0
        
        # Anomaly thresholds
        self.thresholds = {
            "loss_spike": 2.0,     # If loss > 2.0 * moving_avg
            "grad_explosion": 5.0, # If grad_norm > 5.0
            "perplexity_high": 100.0 # If perplexity > 100
        }
        self.moving_avg_loss = 0.0

    def update(self, metrics):
        """
        Updates the current state with new metrics and checks for anomalies.
        """
        current_time = time.time()
        self.steps_since_save += 1
        
        # Calculate derived metrics
        loss = metrics.get("loss", 0.0)
        try:
            perplexity = math.exp(loss) if loss < 20 else float('inf')
        except OverflowError:
            perplexity = float('inf')
            
        # Update latest metrics
        self.latest_metrics.update(metrics)
        self.latest_metrics["timestamp"] = current_time
        self.latest_metrics["perplexity"] = perplexity
        self.latest_metrics["anomalies"] = self._detect_anomalies(loss, metrics.get("grad_norm", 0.0), perplexity)
        
        # Update moving average for loss
        if self.moving_avg_loss == 0.0:
            self.moving_avg_loss = loss
        else:
            self.moving_avg_loss = 0.95 * self.moving_avg_loss + 0.05 * loss

        # Add to history (keep last 1000 in memory for GUI graphs if needed)
        self.history.append(self.latest_metrics.copy())
        if len(self.history) > 1000:
            self.history.pop(0)
            
        # Check if we should save to disk
        if (current_time - self.last_save_time >= self.save_interval_seconds) or \
           (self.steps_since_save >= self.save_interval_steps):
            self.save_snapshot()

    def _detect_anomalies(self, loss, grad_norm, perplexity):
        anomalies = []
        
        # Loss Spike
        if self.moving_avg_loss > 0 and loss > self.thresholds["loss_spike"] * self.moving_avg_loss:
            anomalies.append(f"Loss Spike: {loss:.4f} (Avg: {self.moving_avg_loss:.4f})")
            
        # Gradient Explosion
        if grad_norm > self.thresholds["grad_explosion"]:
            anomalies.append(f"Grad Explosion: {grad_norm:.4f}")
            
        # High Perplexity (Gibberish)
        if perplexity > self.thresholds["perplexity_high"]:
            anomalies.append(f"High Perplexity: {perplexity:.2f}")
            
        return anomalies

    def save_snapshot(self):
        """
        Saves the current history snapshot to a JSON file.
        """
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"diag_{timestamp_str}.json"
        filepath = os.path.join(self.diagnostics_dir, filename)
        
        snapshot = {
            "saved_at": timestamp_str,
            "metrics": self.latest_metrics
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(snapshot, f, indent=2)
            
            self.last_save_time = time.time()
            self.steps_since_save = 0
            # print(f"Diagnostics saved to {filename}") # Optional logging
        except Exception as e:
            print(f"Failed to save diagnostics: {e}")

    def get_latest(self):
        return self.latest_metrics
