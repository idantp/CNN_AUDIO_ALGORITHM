from torch import nn
import torch.nn.functional as F

FILTER_SIZE = 5

LINEAR_L_1 = 4800
LINEAR_L_2 = 1000
LINEAR_L_3 = 100
LABELS_NUM = 30
PADDING_SIZE = 2
FILTER_STRIDES = 1
POOL_STRIDES = 2
KERNEL_SIZE = 2


class ConvolutionNN(nn.Module):
    def __init__(self):
        super(ConvolutionNN, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=FILTER_SIZE, stride=FILTER_STRIDES, padding=PADDING_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=KERNEL_SIZE, stride=POOL_STRIDES))
        self.second_layer = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=FILTER_SIZE, stride=FILTER_STRIDES, padding=PADDING_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=KERNEL_SIZE, stride=POOL_STRIDES))
        self.third_layer = nn.Sequential(
            nn.Conv2d(20, 20, kernel_size=FILTER_SIZE, stride=FILTER_STRIDES, padding=PADDING_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=KERNEL_SIZE, stride=POOL_STRIDES))
        self.drop_out = nn.Dropout()
        self.function_one = nn.Linear(LINEAR_L_1, LINEAR_L_2)
        self.function_two = nn.Linear(LINEAR_L_2, LINEAR_L_3)
        self.function_three = nn.Linear(LINEAR_L_3, LABELS_NUM)

    def forward(self, x):
        out = self.first_layer(x)
        out = self.second_layer(out)
        out = self.third_layer(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.function_one(out)
        out = self.function_two(out)
        out = self.function_three(out)
        return F.log_softmax(out)
