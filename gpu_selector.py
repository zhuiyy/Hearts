import torch
import subprocess
import sys

def get_gpu_info():
    """
    Get information about available GPUs using nvidia-smi and torch.
    """
    if not torch.cuda.is_available():
        return []

    gpus = []
    device_count = torch.cuda.device_count()

    # Try to get detailed info from nvidia-smi
    try:
        # Query index, name, utilization.gpu, memory.used, memory.total
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        lines = result.strip().split('\n')
        for line in lines:
            idx, name, util, mem_used, mem_total = [x.strip() for x in line.split(',')]
            gpus.append({
                'index': int(idx),
                'name': name,
                'utilization': float(util),
                'memory_used': float(mem_used),
                'memory_total': float(mem_total),
                'free_memory': float(mem_total) - float(mem_used)
            })
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to torch properties if nvidia-smi fails
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            # torch doesn't give current utilization easily without extra libs
            gpus.append({
                'index': i,
                'name': props.name,
                'utilization': -1, # Unknown
                'memory_used': -1,
                'memory_total': props.total_memory / (1024**2), # MB
                'free_memory': -1
            })

    return gpus

def select_device():
    """
    List GPUs and ask user to select one. Returns the selected torch device.
    """
    if not torch.cuda.is_available():
        print("No CUDA devices found. Using CPU.")
        return torch.device("cpu")

    gpus = get_gpu_info()
    
    print("\n" + "="*60)
    print(f"{'Idx':<5} {'Name':<25} {'Util(%)':<10} {'Mem(MB)':<15} {'Status'}")
    print("-" * 60)
    
    for gpu in gpus:
        status = "Idle" if gpu['utilization'] < 5 else "Busy"
        mem_info = f"{gpu['memory_used']:.0f}/{gpu['memory_total']:.0f}"
        print(f"{gpu['index']:<5} {gpu['name']:<25} {gpu['utilization']:<10} {mem_info:<15} {status}")
    print("="*60 + "\n")

    while True:
        try:
            choice = input(f"Select GPU index (0-{len(gpus)-1}) or 'cpu': ").strip().lower()
            if choice == 'cpu':
                return torch.device("cpu")
            
            idx = int(choice)
            if 0 <= idx < len(gpus):
                print(f"Selected GPU {idx}: {gpus[idx]['name']}")
                return torch.device(f"cuda:{idx}")
            else:
                print("Invalid index.")
        except ValueError:
            print("Invalid input.")

if __name__ == "__main__":
    device = select_device()
    print(f"Final selection: {device}")
