import time
import psutil
import GPUtil
from datetime import datetime

def monitor_resources(interval=5, duration=15):
    print(f"开始监控资源使用情况 (间隔: {interval}秒, 持续: {duration}秒)")
    print("=" * 80)
    
    start_time = time.time()
    end_time = start_time + duration
    
    cpu_values = []
    memory_values = []
    gpu_load_values = []
    gpu_memory_values = []
    
    while time.time() < end_time:
        current_time = datetime.now().strftime("%H:%M:%S")
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_load = gpu.load * 100
            gpu_memory_used = gpu.memoryUsed
            gpu_memory_total = gpu.memoryTotal
            gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
            gpu_temp = gpu.temperature
        else:
            gpu_load = 0
            gpu_memory_used = 0
            gpu_memory_total = 0
            gpu_memory_percent = 0
            gpu_temp = 0
        
        cpu_values.append(cpu_percent)
        memory_values.append(memory.percent)
        gpu_load_values.append(gpu_load)
        gpu_memory_values.append(gpu_memory_percent)
        
        print(f"[{current_time}] CPU: {cpu_percent:5.1f}% | "
              f"内存: {memory.percent:5.1f}% ({memory.used/1024/1024/1024:5.2f}GB/{memory.total/1024/1024/1024:5.2f}GB) | "
              f"GPU: {gpu_load:5.1f}% | "
              f"GPU显存: {gpu_memory_percent:5.1f}% ({gpu_memory_used/1024:5.1f}GB/{gpu_memory_total/1024:5.1f}GB) | "
              f"GPU温度: {gpu_temp:3.0f}°C")
        
        time.sleep(interval)
    
    print("=" * 80)
    print("监控结束")
    print("\n" + "=" * 80)
    print("资源使用统计和优化建议")
    print("=" * 80)
    
    avg_cpu = sum(cpu_values) / len(cpu_values)
    avg_memory = sum(memory_values) / len(memory_values)
    avg_gpu_load = sum(gpu_load_values) / len(gpu_load_values)
    avg_gpu_memory = sum(gpu_memory_values) / len(gpu_memory_values)
    
    print(f"\n平均CPU使用率: {avg_cpu:.1f}%")
    print(f"平均内存使用率: {avg_memory:.1f}%")
    print(f"平均GPU使用率: {avg_gpu_load:.1f}%")
    print(f"平均GPU显存使用率: {avg_gpu_memory:.1f}%")
    
    print("\n优化建议:")
    print("-" * 80)
    
    if avg_cpu < 50:
        print("✓ CPU使用率较低 (<50%)，可以增加batch_size或num_workers")
    elif avg_cpu < 80:
        print("○ CPU使用率适中 (50-80%)，当前参数合理")
    else:
        print("✗ CPU使用率较高 (>80%)，建议减小batch_size")
    
    if avg_memory < 70:
        print("✓ 内存使用率较低 (<70%)，可以增加batch_size")
    elif avg_memory < 85:
        print("○ 内存使用率适中 (70-85%)，当前参数合理")
    else:
        print("✗ 内存使用率较高 (>85%)，建议减小batch_size")
    
    if avg_gpu_load < 60:
        print("✓ GPU使用率较低 (<60%)，可以增加batch_size或num_beams")
    elif avg_gpu_load < 85:
        print("○ GPU使用率适中 (60-85%)，当前参数合理")
    else:
        print("✗ GPU使用率较高 (>85%)，建议减小batch_size或num_beams")
    
    if avg_gpu_memory < 70:
        print("✓ GPU显存使用率较低 (<70%)，可以增加batch_size")
    elif avg_gpu_memory < 85:
        print("○ GPU显存使用率适中 (70-85%)，当前参数合理")
    else:
        print("✗ GPU显存使用率较高 (>85%)，建议减小batch_size")
    
    print("\n具体参数调整建议:")
    print("-" * 80)
    
    if avg_cpu < 50 and avg_gpu_load < 60 and avg_gpu_memory < 70:
        print("当前资源利用率较低，建议：")
        print("  - batch_size: 1 → 2 (增加一倍)")
        print("  - num_workers: 0 → 2 (增加数据加载速度)")
        print("  - num_beams: 8 → 10 (略微增加)")
    elif avg_cpu > 80 or avg_memory > 85 or avg_gpu_memory > 85:
        print("当前资源利用率较高，建议：")
        print("  - batch_size: 1 → 保持不变或减小")
        print("  - num_workers: 0 (保持不变)")
        print("  - num_beams: 8 → 6 (减小)")
    else:
        print("当前资源利用率适中，建议保持当前参数不变")
    
    print("=" * 80)

if __name__ == "__main__":
    monitor_resources(interval=3, duration=15)
