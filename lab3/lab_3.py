import numpy as np
import torch

def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups = 1, dilation = 1):
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, in_channels, kernel_height, kernel_width = weight.shape

    out_height = (in_height - 1) * stride - 2 * padding + kernel_height + output_padding
    out_width = (in_width - 1) * stride - 2 * padding + kernel_width + output_padding

    output = np.zeros((batch_size, out_channels, out_height, out_width))

    for b in range (batch_size):
      for c in range(out_channels):
        for i in range(out_height):
          for j in range(out_width):
            for k in range(in_channels):
              for s in range(kernel_height):
                for t in range(kernel_width):
                    h_idx = i + padding - dilation * s
                    w_idx = j + padding - dilation * t
                    if h_idx >= 0 and w_idx >= 0 and h_idx < in_height * stride and w_idx < in_width * stride and (h_idx % stride == 0) and (w_idx % stride == 0):
                      h_idx //= stride
                      w_idx //= stride
                      output[b, c, i, j] += input[b, k, h_idx, w_idx] * weight[c, k, s, t]

    # Add bias if provided
    if bias is not None:
        output[b, c, :, :] += bias[c]

    return output

# Тест 1 
input_tensor = torch.randn(1, 1, 3, 3)
weight_tensor = torch.randn(1, 1, 3, 3)  
output_tensor_custom = conv_transpose2d(input_tensor.numpy(), weight_tensor.numpy())
print("Custom function result:\n", output_tensor_custom)
output_tensor_torch = torch.nn.functional.conv_transpose2d(input_tensor, weight_tensor).numpy()
print("\nPyTorch function result:\n", output_tensor_torch)
print("\nDo results match:", np.allclose(output_tensor_custom, output_tensor_torch))

# Тест 2
input_tensor = torch.randn(1, 1, 3, 3)
weight_tensor = torch.randn(1, 1, 3, 3)  
output_tensor_custom = conv_transpose2d(input_tensor.numpy(), weight_tensor.numpy())
output_tensor_torch = torch.nn.functional.conv_transpose2d(input_tensor, weight_tensor).numpy()
print("\nDo results match:", np.allclose(output_tensor_custom, output_tensor_torch))

# Тест 3
input_tensor = torch.randn(1, 1, 3, 3)
weight_tensor = torch.randn(1, 1, 3, 3)  
output_tensor_custom = conv_transpose2d(input_tensor.numpy(), weight_tensor.numpy())
output_tensor_torch = torch.nn.functional.conv_transpose2d(input_tensor, weight_tensor).numpy()
print("\nDo results match:", np.allclose(output_tensor_custom, output_tensor_torch))