import torch
import sys
import os

sys.path.append(os.path.abspath('../'))
import src.utils as utils

# Log for cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Working with {device}")

class ResBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, norm_func, kernel_size=3, stride=1):
        super(ResBlock, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channel, in_channel, kernel_size, stride=1, padding=1)
        self.norm1 = norm_func(in_channel)
        self.activation1 = torch.nn.ReLU()
        
        self.conv2 = torch.nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=1)
        self.norm2 = norm_func(out_channel)
        self.activation2 = torch.nn.ReLU()
        
        self.project = True if (in_channel != out_channel) or (stride != 1) else False
        if self.project:
            self.conv_project = torch.nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=1)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x += self.conv_project(res) if self.project else res
        x = self.activation2(x)
        
        return x
        

class Resnet(torch.nn.Module):
    def __init__(self, n, r, norm_func = torch.nn.BatchNorm2d):
        super(Resnet, self).__init__()

        self.norm = norm_func
        
        #Input
        self.input_layer = []
        self.input_layer.append(torch.nn.Conv2d(3, 16, 3, 1, padding=1))
        self.input_layer.append(self.norm(16))
        self.input_layer.append(torch.nn.ReLU())
        self.input_layer = torch.nn.Sequential(*self.input_layer)
        
        # Layer1
        self.hidden_layer1 = []
        for i in range(n):
            self.hidden_layer1.append(ResBlock(16, 16, self.norm))
        self.hidden_layer1 = torch.nn.Sequential(*self.hidden_layer1)
        
        # Layer2
        self.hidden_layer2 = []
        self.hidden_layer2.append(ResBlock(16, 32, self.norm, stride = 2))
        for i in range(n-1):
            self.hidden_layer2.append(ResBlock(32, 32, self.norm))
        self.hidden_layer2 = torch.nn.Sequential(*self.hidden_layer2)
        
        # Layer3
        self.hidden_layer3 = []
        self.hidden_layer3.append(ResBlock(32, 64, self.norm, stride = 2))
        for i in range(n-1):
            self.hidden_layer3.append(ResBlock(64, 64, self.norm))
        self.hidden_layer3 = torch.nn.Sequential(*self.hidden_layer3)
            
        # Pool Layer
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.flatten = torch.nn.Flatten()
        
        # Output Layer
        self.output_layer = []
        self.output_layer.append(torch.nn.Linear(64, r))
        self.output_layer.append(torch.nn.Softmax(r))
        self.output_layer = torch.nn.Sequential(*self.output_layer)

        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.output_layer(x)
        
        return x


def set_seed(rand_seed):
    torch.manual_seed(rand_seed)