import torch

import torch

def custom_conv2d(input_tensor, weight, bias=None, stride=1, padding=0):

    if input_tensor.dim() != 4 or weight.dim() != 4:
        raise ValueError("Инпут и веса должны быть 4-мерными тензорами.")

    batch_size, in_channels, input_height, input_width = input_tensor.size()
    out_channels, _, kernel_height, kernel_width = weight.size()

    if in_channels != weight.size(1):
        raise ValueError("Количество входных каналов должно соответствовать количеству каналов весов.")

    output_height = (input_height + 2 * padding - kernel_height) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1

    padded_input = torch.nn.functional.pad(input_tensor, (padding, padding, padding, padding))

    output_tensor = torch.zeros(batch_size, out_channels, output_height, output_width)

    for b in range(batch_size):
        for c_out in range(out_channels):
            for h_out in range(output_height):
                for w_out in range(output_width):
                    h_start = h_out * stride
                    h_end = h_start + kernel_height
                    w_start = w_out * stride
                    w_end = w_start + kernel_width

                    input_patch = padded_input[b, :, h_start:h_end, w_start:w_end]

                    output_tensor[b, c_out, h_out, w_out] = torch.sum(input_patch * weight[c_out]) + (bias[c_out] if bias is not None else 0)

    return output_tensor

