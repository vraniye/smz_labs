import unittest
import torch
import torch.nn.functional as F
from fu_custom_conv3d import custom_conv3d

class TestConv3d(unittest.TestCase):
    def setUp(self):
        # параметры для тестов
        self.batch_size = 2
        self.in_channels = 3
        self.input_depth = 4
        self.input_height = 5
        self.input_width = 5
        self.out_channels = 4
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1

        self.input_tensor = torch.randn(self.batch_size, self.in_channels, self.input_depth, self.input_height, self.input_width)
        self.weight = torch.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size)
        self.bias = torch.randn(self.out_channels)

    def test_custom_conv3d(self):
        custom_output = custom_conv3d(self.input_tensor, self.weight, self.bias, stride=self.stride, padding=self.padding)
        torch_output = F.conv3d(self.input_tensor, self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
        self.assertEqual(custom_output.size(), torch_output.size())
        tolerance = 1e-5
        self.assertTrue(torch.allclose(custom_output, torch_output, rtol=tolerance, atol=tolerance))

    def test_custom_conv3d_no_bias(self):
        custom_output = custom_conv3d(self.input_tensor, self.weight, stride=self.stride, padding=self.padding)
        torch_output = F.conv3d(self.input_tensor, self.weight, stride=self.stride, padding=self.padding)
        self.assertEqual(custom_output.size(), torch_output.size())
        tolerance = 1e-5
        self.assertTrue(torch.allclose(custom_output, torch_output, rtol=tolerance, atol=tolerance))

    def test_custom_conv3d_large_input(self):
        large_input_tensor = torch.randn(5, self.in_channels, 10, 10, 10)
        custom_output = custom_conv3d(large_input_tensor, self.weight, self.bias, stride=self.stride, padding=self.padding)
        torch_output = F.conv3d(large_input_tensor, self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
        self.assertEqual(custom_output.size(), torch_output.size())
        tolerance = 1e-5
        self.assertTrue(torch.allclose(custom_output, torch_output, rtol=tolerance, atol=tolerance))

if __name__ == '__main__':
    unittest.main()