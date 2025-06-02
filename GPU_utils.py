import torch
import time
import subprocess
import sys
import os

def print_gpu_utilization():
    """In thống kê sử dụng GPU"""
    if torch.cuda.is_available():
        print(f"\nGPU utilization stats:")
        print(f"- Device: {torch.cuda.get_device_name(0)}")
        print(f"- Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"- Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        print(f"- Is GPU being used: {'Yes' if torch.cuda.memory_allocated(0) > 0 else 'No'}")
        
        # Thực thi lệnh nvidia-smi nếu có thể
        try:
            if sys.platform != 'win32':
                result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
                print("\nnvidia-smi output:")
                print(result.stdout.decode('utf-8'))
        except:
            pass
    else:
        print("CUDA not available")

def force_cuda_device_init():
    """Khởi tạo GPU một cách rõ ràng"""
    if torch.cuda.is_available():
        # Thiết lập biến môi trường
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.cuda.set_device(0)
        
        # Tạo tensor trên GPU và thực hiện phép tính đơn giản
        x = torch.rand(1000, 1000, device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()  # Đảm bảo xử lý xong trên GPU
        
        del x, y  # Giải phóng bộ nhớ
        torch.cuda.empty_cache()
        
        return True
    return False

class GPUMemoryTracker:
    """Theo dõi bộ nhớ GPU trong quá trình train"""
    def __init__(self):
        self.enabled = torch.cuda.is_available()
        self.start_allocated = 0
        self.start_reserved = 0
        
    def start(self):
        """Bắt đầu theo dõi"""
        if self.enabled:
            torch.cuda.synchronize()
            self.start_allocated = torch.cuda.memory_allocated()
            self.start_reserved = torch.cuda.memory_reserved()
            print(f"Starting memory tracking - Allocated: {self.start_allocated/1024**2:.2f}MB")
    
    def end(self):
        """Kết thúc theo dõi và báo cáo"""
        if self.enabled:
            torch.cuda.synchronize()
            end_allocated = torch.cuda.memory_allocated()
            end_reserved = torch.cuda.memory_reserved()
            
            print(f"Memory change - Allocated: {(end_allocated-self.start_allocated)/1024**2:.2f}MB, "
                  f"Reserved: {(end_reserved-self.start_reserved)/1024**2:.2f}MB")