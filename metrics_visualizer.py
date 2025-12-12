import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


class MetricsVisualizer:
    """Compare and visualize metrics across multiple model architectures."""
    
    def __init__(self, metrics_dir="./metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.models_data = {}

    # ------------------ Loading helpers ------------------
        
    def load_all_metrics(self):
        """Load all metric files from the directory."""
        for file in self.metrics_dir.glob("*_metrics.json"):
            with open(file, 'r') as f:
                data = json.load(f)
                model_name = data['model_info']['model_name']
                self.models_data[model_name] = data
        
        print(f"Loaded {len(self.models_data)} model runs: {list(self.models_data.keys())}")
        return self.models_data

    def _ensure_loaded(self) -> bool:
        """Make sure metrics are loaded; return False if nothing found."""
        if not self.models_data:
            self.load_all_metrics()
        if not self.models_data:
            print("No metrics files found!")
            return False
        return True

    def _iter_selected_models(self, model_names=None):
        """
        Yield (name, data) pairs.
        - If model_names is None: all models.
        - If model_names is a list: only those models.
        - Warns about any models not found.
        """
        if model_names is None:
            return self.models_data.items()
        
        # Ensure model_names is a list
        if isinstance(model_names, str):
            model_names = [model_names]
        
        # Check for missing models
        missing = [name for name in model_names if name not in self.models_data]
        if missing:
            print(f"Model(s) {missing} not found. "
                  f"Available models: {list(self.models_data.keys())}")
        
        # Yield only found models
        return [(name, self.models_data[name]) for name in model_names 
                if name in self.models_data]

    # ------------------ Individual plot functions ------------------

    def plot_training_loss(self, model_names=None, ax=None):
        """1. Training Loss Evolution."""
        if not self._ensure_loaded():
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        
        for name, data in self._iter_selected_models(model_names):
            losses = data['batch_metrics']['batch_losses']
            # Smooth with moving average
            window = min(50, max(1, len(losses) // 10)) or 1
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=name, alpha=0.8)
        
        ax.set_xlabel('Batch (smoothed)')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_test_accuracy(self, model_names=None, ax=None):
        """2. Test Accuracy by Epoch."""
        if not self._ensure_loaded():
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        
        for name, data in self._iter_selected_models(model_names):
            epochs = data['epoch_metrics']['epoch']
            acc = data['epoch_metrics']['test_accuracy']
            ax.plot(epochs, acc, marker='o', label=name)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Accuracy [%]')
        ax.set_title('Test Accuracy Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_gpu_memory_usage(self, model_names=None, ax=None):
        """3. GPU Memory Usage over batches."""
        if not self._ensure_loaded():
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        
        any_plotted = False
        for name, data in self._iter_selected_models(model_names):
            if 'gpu_memory_allocated_mb' in data['batch_metrics']:
                mem = data['batch_metrics']['gpu_memory_allocated_mb']
                ax.plot(mem, label=name, alpha=0.7)
                any_plotted = True
        
        ax.set_xlabel('Batch')
        ax.set_ylabel('Allocated GPU Memory [MB]')
        ax.set_title('GPU Memory Usage')
        if any_plotted:
            ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_throughput(self, model_names=None, ax=None):
        """4. Average throughput per model."""
        if not self._ensure_loaded():
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        
        names_list = []
        throughputs = []
        for name, data in self._iter_selected_models(model_names):
            names_list.append(name)
            avg_throughput = np.mean(data['epoch_metrics']['throughput_mean'])
            throughputs.append(avg_throughput)
        
        ax.bar(names_list, throughputs)
        ax.set_ylabel('Samples per second [1/s]')
        ax.set_title('Average Throughput')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        return ax

    def plot_model_size_vs_accuracy(self, model_names=None, ax=None):
        """5. Model Size vs Final Accuracy."""
        if not self._ensure_loaded():
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        
        for name, data in self._iter_selected_models(model_names):
            params = data['model_info']['total_params'] / 1e6  # in millions
            final_acc = data['epoch_metrics']['test_accuracy'][-1]
            ax.scatter(params, final_acc, s=100, label=name)
            ax.text(params, final_acc, f' {name}', fontsize=9)
        
        ax.set_xlabel('Parameters (millions)')
        ax.set_ylabel('Final Test Accuracy [%]')
        ax.set_title('Model Size vs Performance')
        ax.grid(True, alpha=0.3)
        return ax

    def plot_training_time(self, model_names=None, ax=None):
        """6. Total training time per model."""
        if not self._ensure_loaded():
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        
        names_list = []
        total_times = []
        for name, data in self._iter_selected_models(model_names):
            names_list.append(name)
            total_time = sum(data['epoch_metrics']['epoch_time_sec'])
            total_times.append(total_time)
        
        ax.bar(names_list, total_times)
        ax.set_ylabel('Total Training Time [s]')
        ax.set_title('Total Training Time')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        return ax

    # ------------------ Combined comparison figure ------------------

    def plot_comparison(self, save_path="model_comparison.png", model_names=None):
        """
        Generate comprehensive comparison plots in a 2x3 grid.
        - By default, compares all models.
        - If model_names is given (string or list), only plots those models.
        """
        if not self._ensure_loaded():
            return
        
        fig = plt.figure(figsize=(16, 10),dpi=200)
        
        ax1 = plt.subplot(2, 2, 1)
        self.plot_training_loss(model_names=model_names, ax=ax1)
        
        ax2 = plt.subplot(2, 2, 2)
        self.plot_test_accuracy(model_names=model_names, ax=ax2)
        
        ax3 = plt.subplot(2, 2, 3)
        self.plot_gpu_memory_usage(model_names=model_names, ax=ax3)
        
        #ax4 = plt.subplot(2, 3, 4)
        #self.plot_throughput(model_names=model_names, ax=ax4)
        
        ax5 = plt.subplot(2, 2, 4)
        self.plot_model_size_vs_accuracy(model_names=model_names, ax=ax5)
        
        #ax6 = plt.subplot(2, 3, 6)
        #self.plot_training_time(model_names=model_names, ax=ax6)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
        plt.show()
        plt.close()

    # ------------------ Summary table (unchanged) ------------------

    def print_summary_table(self):
        """Print comparison table of key metrics."""
        if not self.models_data:
            self.load_all_metrics()
        
        print("\n" + "="*100)
        print(f"{'Model':<15} {'Params [M]':<12} {'Final Acc %':<12} {'Best Acc %':<12} "
              f"{'Time (s)':<12} {'Peak GPU [MB]':<15}")
        print("="*100)
        
        for model_name, data in self.models_data.items():
            params = data['model_info']['total_params'] / 1e6
            final_acc = data['epoch_metrics']['test_accuracy'][-1]
            best_acc = max(data['epoch_metrics']['test_accuracy'])
            total_time = sum(data['epoch_metrics']['epoch_time_sec'])
            
            if 'gpu_memory_peak_mb' in data['epoch_metrics']:
                peak_gpu = max(data['epoch_metrics']['gpu_memory_peak_mb'])
                gpu_str = f"{peak_gpu:.1f}"
            else:
                gpu_str = "N/A"
            
            print(f"{model_name:<15} {params:<12.2f} {final_acc:<12.2f} {best_acc:<12.2f} "
                  f"{total_time:<12.1f} {gpu_str:<15}")
        
        print("="*100 + "\n")


if __name__ == "__main__":
    # Example usage
    viz = MetricsVisualizer()
    viz.load_all_metrics()
    viz.print_summary_table()
    
    # All models (default):
    viz.plot_comparison()
    
    # Single model (string):
    # viz.plot_comparison(model_names="MyModel")
    
    # Multiple specific models (list):
    # viz.plot_comparison(model_names=["Model1", "Model2", "Model3"])
    
    # Individual plot examples:
    # viz.plot_training_loss(model_names=["Model1", "Model2"])
    # viz.plot_test_accuracy(model_names="Model1")  # Also accepts string
