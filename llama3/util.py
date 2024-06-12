import psutil
import platform
cpu_model = platform.processor()
memory_info = psutil.virtual_memory()
total_memory_gb = memory_info.total / (1024 ** 3)
memory_usage_percent = memory_info.percent
cpu_threads = psutil.cpu_count(logical=True)
cpu_usage = psutil.cpu_percent(interval=1)

print(f"CPU: {cpu_model}")
print(f"Memory: {total_memory_gb:.2f} GB")
print(f"Memory usage: {memory_usage_percent:.2f}%")
print(f"CPU threads: {cpu_threads}")
print(f"CPU usage: {cpu_usage}%")

import GPUtil
GPUtil.showUtilization()
gpus = GPUtil.getGPUs()
for gpu in gpus:
    print(f"GPU: {gpu.name}")
    print(f"Memory Total: {gpu.memoryTotal/1024:.2f} GB")
