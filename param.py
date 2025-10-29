import torch
import time
from thop import profile
from models.Models import DPCD

# 检查 CUDA 是否可用
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建模型并移动到指定设备
net = DPCD().to(device)

# 创建输入数据并移动到指定设备
input = torch.randn(1, 3, 256, 256).to(device)

# 计算 FLOPs 和参数数量
flops, params = profile(net, inputs=(input, input))
print('flops: ', flops, 'params: ', params)

# 测量推理时间
with torch.no_grad():
    start_time = time.time()
    output = net(input, input)
    end_time = time.time()
    inference_time = end_time - start_time

print(f'Inference time: {inference_time:.4f} seconds')
