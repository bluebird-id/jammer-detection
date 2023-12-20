from collections import Counter
has_pynvml = True
try:
    from pynvml import *
except ImportError:
    has_pynvml = False

def get_gpu_count():
    """Get the number of GPU devices available for the current process (0 if no GPU)"""
    if not has_pynvml:
        return 0
    try:
        nvmlInit()
    except NVMLError:
        return 0
    return nvmlDeviceGetCount()


def print_gpu_info():
    """If there are GPUs it will print the name of each one which the number of devices"""
    device_count = get_gpu_count()
    if device_count == 0:
        return
    devices_count = Counter()
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        device_name = nvmlDeviceGetName(handle).decode("utf-8")
        devices_count[device_name] += 1

    counts = [ f"{device_name} ({devices_count[device_name]})" for device_name in devices_count ]
    print(f"GPUs: {', '.join(counts)}")