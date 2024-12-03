import torch
import torch
x = torch.rand(5, 3)
print(x)

print(torch.cuda.is_available())  # Should return True if everything is set up correctly
print(torch.cuda.get_device_name(0))  # Should print the name of your GPU (e.g., 'NVIDIA GeForce RTX 4050')
