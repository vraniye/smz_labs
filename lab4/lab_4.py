import numpy as np
import torch

def custom_upsampling(input_tensor, size):

    N, C, H, W = input_tensor.shape
    new_H, new_W = size

    scale_factor_H = new_H / H
    scale_factor_W = new_W / W

    output_np = np.empty((N, C, new_H, new_W))

    for n in range(N):
        for c in range(C):
            for i in range(new_H):
                for j in range(new_W):
                    x = (i + 0.5) / scale_factor_H - 0.5
                    y = (j + 0.5) / scale_factor_W - 0.5
                    x0 = int(np.floor(x))
                    x1 = min(x0 + 1, H - 1)
                    y0 = int(np.floor(y))
                    y1 = min(y0 + 1, W - 1)
                    output_np[n, c, i, j] = (input_tensor[n, c, x0, y0] * (x1 - x) * (y1 - y) +
                                             input_tensor[n, c, x0, y1] * (x1 - x) * (y - y0) +
                                             input_tensor[n, c, x1, y0] * (x - x0) * (y1 - y) +
                                             input_tensor[n, c, x1, y1] * (x - x0) * (y - y0))

    return torch.from_numpy(output_np)


# Тест 1 
input_tensor = torch.randn(1, 1, 4, 4)
output_size = (1, 1)  
output_tensor_custom = custom_upsampling(input_tensor, output_size)
print("Custom function result:\n", output_tensor_custom)
output_tensor_torch = torch.nn.functional.upsample(input_tensor, output_size, mode = 'bilinear')
print("\nPyTorch function result:\n", output_tensor_torch)
print("\nDo results match:", np.allclose(output_tensor_custom, output_tensor_torch))

# Тест 2
input_tensor = torch.randn(1, 1, 4, 4)
output_size = (1, 1)  
output_tensor_custom = custom_upsampling(input_tensor, output_size)
output_tensor_torch = torch.nn.functional.upsample(input_tensor, output_size, mode = 'bilinear')
print("\nDo results match:", np.allclose(output_tensor_custom, output_tensor_torch))

# Тест 3
input_tensor = torch.randn(1, 1, 4, 4)
output_size = (1, 1)  
output_tensor_custom = custom_upsampling(input_tensor, output_size)
output_tensor_torch = torch.nn.functional.upsample(input_tensor, output_size, mode = 'bilinear')
print("\nDo results match:", np.allclose(output_tensor_custom, output_tensor_torch))