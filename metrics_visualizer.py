import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


class MetricsVisualizer:
    """Compare and visualize metrics across multiple model architectures."""
    
    def __init__(self, metrics_dir="./metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.models_data = {}
        
    def load_all_metrics(self):
        """Load all metric files from the directory."""
        for file in self.metrics_dir.glob("*_metrics.json"):
            with open(file, 'r') as f:
                data = json.load(f)
                model_name = data['model_info']['model_name']
                self.models_data[model_name] = data
        
        print(f"Loaded {len(self.models_data)} model runs: {list(self.models_data.keys())}")
        return self.models_data
    
    def plot_comparison(self, save_path="model_comparison.png"):
        """Generate comprehensive comparison plots."""
        if not self.models_data:
            self.load_all_metrics()
        
        if not self.models_data:
            print("No metrics files found!")
            return
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Training Loss Evolution
        ax1 = plt.subplot(2, 3, 1)
        for model_name, data in self.models_data.items():
            epochs = data['batch_metrics']['epochs_list']
            losses = data['batch_metrics']['batch_losses']
            # Smooth with moving average
            window = min(50, len(losses) // 10) or 1
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax1.plot(smoothed, label=model_name, alpha=0.8)
        ax1.set_xlabel('Batch (smoothed)')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Test Accuracy by Epoch
        ax2 = plt.subplot(2, 3, 2)
        for model_name, data in self.models_data.items():
            epochs = data['epoch_metrics']['epoch']
            acc = data['epoch_metrics']['test_accuracy']
            ax2.plot(epochs, acc, marker='o', label=model_name)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Test Accuracy (%)')
        ax2.set_title('Test Accuracy Progress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. GPU Memory Usage
        ax3 = plt.subplot(2, 3, 3)
        for model_name, data in self.models_data.items():
            if 'gpu_memory_allocated_mb' in data['batch_metrics']:
                mem = data['batch_metrics']['gpu_memory_allocated_mb']
                ax3.plot(mem, label=model_name, alpha=0.7)
        ax3.set_xlabel('Batch')
        ax3.set_ylabel('GPU Memory (MB)')
        ax3.set_title('GPU Memory Usage')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Throughput Comparison
        ax4 = plt.subplot(2, 3, 4)
        model_names = []
        throughputs = []
        for model_name, data in self.models_data.items():
            model_names.append(model_name)
            avg_throughput = np.mean(data['epoch_metrics']['throughput_mean'])
            throughputs.append(avg_throughput)
        ax4.bar(model_names, throughputs, color='steelblue')
        ax4.set_ylabel('Samples/sec')
        ax4.set_title('Average Throughput')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Model Size vs Accuracy
        ax5 = plt.subplot(2, 3, 5)
        for model_name, data in self.models_data.items():
            params = data['model_info']['total_params'] / 1e6  # in millions
            final_acc = data['epoch_metrics']['test_accuracy'][-1]
            ax5.scatter(params, final_acc, s=100, label=model_name)
            ax5.text(params, final_acc, f' {model_name}', fontsize=9)
        ax5.set_xlabel('Parameters (millions)')
        ax5.set_ylabel('Final Test Accuracy (%)')
        ax5.set_title('Model Size vs Performance')
        ax5.grid(True, alpha=0.3)
        
        # 6. Training Time Comparison
        ax6 = plt.subplot(2, 3, 6)
        model_names = []
        total_times = []
        for model_name, data in self.models_data.items():
            model_names.append(model_name)
            total_time = sum(data['epoch_metrics']['epoch_time_sec'])
            total_times.append(total_time)
        ax6.bar(model_names, total_times, color='coral')
        ax6.set_ylabel('Total Training Time (s)')
        ax6.set_title('Total Training Time')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
        plt.show()
        plt.close()
    
    def print_summary_table(self):
        """Print comparison table of key metrics."""
        if not self.models_data:
            self.load_all_metrics()
        
        print("\n" + "="*100)
        print(f"{'Model':<15} {'Params (M)':<12} {'Final Acc %':<12} {'Best Acc %':<12} "
              f"{'Time (s)':<12} {'Peak GPU (MB)':<15}")
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
    viz.plot_comparison()