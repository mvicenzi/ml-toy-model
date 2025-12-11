import time
import psutil
import torch
from collections import defaultdict
from pathlib import Path
import json

class MetricsMonitor:
    """Lightweight training metrics tracker for comparing model architectures."""
    
    def __init__(self, model_name, save_dir="./metrics"):
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(list)
        
        # Timing
        self.epoch_start_time = None
        self.batch_start_time = None
        
        # GPU tracking
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else None
        
    def on_train_begin(self, model):
        """Call at training start to capture model info."""
        self.metrics['model_name'] = self.model_name
        self.metrics['total_params'] = sum(p.numel() for p in model.parameters())
        self.metrics['trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.metrics['model_size_mb'] = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        print(f"\n{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Total params: {self.metrics['total_params']:,}")
        print(f"Trainable params: {self.metrics['trainable_params']:,}")
        print(f"Model size: {self.metrics['model_size_mb']:.2f} MB")
        print(f"{'='*60}\n")
        
    def on_epoch_begin(self, epoch):
        """Call at start of each epoch."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        
    def on_batch_begin(self):
        """Call at start of each batch."""
        self.batch_start_time = time.time()
        
    def on_batch_end(self, batch_idx, loss, batch_size):
        """Call after each batch with loss value."""
        batch_time = time.time() - self.batch_start_time
        
        # Core metrics
        self.metrics['batch_losses'].append(loss)
        self.metrics['batch_times'].append(batch_time)
        self.metrics['epochs_list'].append(self.current_epoch)
        self.metrics['batch_indices'].append(batch_idx)
        
        # Throughput
        samples_per_sec = batch_size / batch_time if batch_time > 0 else 0
        self.metrics['throughput'].append(samples_per_sec)
        
        # Memory metrics
        if self.device is not None:
            gpu_mem_allocated = torch.cuda.memory_allocated(self.device) / (1024**2)
            gpu_mem_reserved = torch.cuda.memory_reserved(self.device) / (1024**2)
            self.metrics['gpu_memory_allocated_mb'].append(gpu_mem_allocated)
            self.metrics['gpu_memory_reserved_mb'].append(gpu_mem_reserved)
        
        # CPU memory
        process = psutil.Process()
        cpu_mem_mb = process.memory_info().rss / (1024**2)
        self.metrics['cpu_memory_mb'].append(cpu_mem_mb)
        
    def on_epoch_end(self, epoch, test_loss, test_acc):
        """Call at end of each epoch with test metrics."""
        epoch_time = time.time() - self.epoch_start_time
        
        # Aggregate epoch metrics
        epoch_start = len([e for e in self.metrics['epochs_list'] if e < epoch])
        epoch_end = len(self.metrics['epochs_list'])
        epoch_losses = self.metrics['batch_losses'][epoch_start:epoch_end]
        
        self.epoch_metrics['epoch'].append(epoch)
        self.epoch_metrics['train_loss_mean'].append(sum(epoch_losses) / len(epoch_losses))
        self.epoch_metrics['train_loss_min'].append(min(epoch_losses))
        self.epoch_metrics['train_loss_max'].append(max(epoch_losses))
        self.epoch_metrics['test_loss'].append(test_loss)
        self.epoch_metrics['test_accuracy'].append(test_acc)
        self.epoch_metrics['epoch_time_sec'].append(epoch_time)
        
        # Memory stats
        if self.device is not None:
            epoch_gpu_mem = self.metrics['gpu_memory_allocated_mb'][epoch_start:epoch_end]
            self.epoch_metrics['gpu_memory_peak_mb'].append(max(epoch_gpu_mem))
            self.epoch_metrics['gpu_memory_mean_mb'].append(sum(epoch_gpu_mem) / len(epoch_gpu_mem))
        
        epoch_cpu_mem = self.metrics['cpu_memory_mb'][epoch_start:epoch_end]
        self.epoch_metrics['cpu_memory_peak_mb'].append(max(epoch_cpu_mem))
        
        # Throughput
        epoch_throughput = self.metrics['throughput'][epoch_start:epoch_end]
        self.epoch_metrics['throughput_mean'].append(sum(epoch_throughput) / len(epoch_throughput))
        
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s | "
              f"Train Loss: {self.epoch_metrics['train_loss_mean'][-1]:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        if self.device is not None:
            print(f"  GPU Memory: {self.epoch_metrics['gpu_memory_peak_mb'][-1]:.1f} MB (peak) | "
                  f"Throughput: {self.epoch_metrics['throughput_mean'][-1]:.1f} samples/sec")
        
    def save(self):
        """Save metrics to JSON file."""
        output_path = self.save_dir / f"{self.model_name}_metrics.json"
        
        # Combine all metrics
        all_metrics = {
            'model_info': {
                'model_name': self.metrics['model_name'],
                'total_params': self.metrics['total_params'],
                'trainable_params': self.metrics['trainable_params'],
                'model_size_mb': self.metrics['model_size_mb'],
            },
            'batch_metrics': {k: v for k, v in self.metrics.items() 
                            if k not in ['model_name', 'total_params', 'trainable_params', 'model_size_mb']},
            'epoch_metrics': dict(self.epoch_metrics),
        }
        
        with open(output_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        print(f"\n✓ Metrics saved to {output_path}")
        return output_path
    
    def print_summary(self):
        """Print final summary statistics."""
        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY - {self.model_name}")
        print(f"{'='*60}")
        print(f"Total epochs: {len(self.epoch_metrics['epoch'])}")
        print(f"Final test accuracy: {self.epoch_metrics['test_accuracy'][-1]:.2f}%")
        print(f"Best test accuracy: {max(self.epoch_metrics['test_accuracy']):.2f}%")
        print(f"Final test loss: {self.epoch_metrics['test_loss'][-1]:.4f}")
        print(f"Total training time: {sum(self.epoch_metrics['epoch_time_sec']):.2f}s")
        print(f"Avg epoch time: {sum(self.epoch_metrics['epoch_time_sec'])/len(self.epoch_metrics['epoch_time_sec']):.2f}s")
        
        if self.device is not None:
            print(f"Peak GPU memory: {max(self.epoch_metrics['gpu_memory_peak_mb']):.1f} MB")
        
        print(f"Peak CPU memory: {max(self.epoch_metrics['cpu_memory_peak_mb']):.1f} MB")
        print(f"Avg throughput: {sum(self.epoch_metrics['throughput_mean'])/len(self.epoch_metrics['throughput_mean']):.1f} samples/sec")
        print(f"{'='*60}\n")