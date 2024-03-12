import torch
import numpy as np
import sys
import os
import time

sys.path.append(os.path.abspath('../'))
import src.utils as utils

# Log for cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Working with {device}")

class ResBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, norm_func, kernel_size=3, stride=1):
        super(ResBlock, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channel, in_channel, kernel_size, stride=1, padding=1, dtype=torch.float64)
        self.norm1 = norm_func(in_channel, dtype=torch.float64)
        self.activation1 = torch.nn.ReLU()
        
        self.conv2 = torch.nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=1, dtype=torch.float64)
        self.norm2 = norm_func(out_channel, dtype=torch.float64)
        self.activation2 = torch.nn.ReLU()
        
        self.project = True if (in_channel != out_channel) or (stride != 1) else False
        if self.project:
            self.conv_project = torch.nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=1, dtype=torch.float64)

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
        self.input_layer.append(torch.nn.Conv2d(3, 16, 3, 1, padding=1, dtype=torch.float64))
        self.input_layer.append(self.norm(16, dtype=torch.float64))
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
        self.output_layer.append(torch.nn.Linear(64, r, dtype=torch.float64))
        self.output_layer.append(torch.nn.Softmax(1))
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

class DataLoader():
    def __init__(self, x_addr, y_addr, batch_size, rand_seed = None, randomize = True, device = device):
        # Finding number of Data Samples
        # self.num_data = len(os.listdir(x_addr))
        self.num_data = 60
        
        # Randomizing if True
        np.random.seed(rand_seed)
        self.order = np.arange(self.num_data)
        
        # Assigning Class Variables
        self.x_addr = x_addr
        self.y_addr = y_addr
        self.batch_size = batch_size
        self.device = device
        self.randomize = randomize
        
    def __iter__(self):
        # Initializing index
        self.ind = 0
        if self.randomize:
            np.random.shuffle(self.order)
        return self
    
    def __next__(self):
        # Checking stop condition
        if self.ind >= self.num_data:
            raise StopIteration
        
        X, Y = [], []
        for i in range(self.batch_size):
            # Loading X
            x = torch.load(os.path.join(self.x_addr, f'{self.order[self.ind]}.pt'))
            X.append(x.permute(2, 0, 1))    # Channel is first dimension
            
            # Loading Y
            y = torch.load(os.path.join(self.y_addr, f'{self.order[self.ind]}.pt'))
            Y.append(y)
            
            # Updating index
            self.ind += 1
            if self.ind == self.num_data:
                break
        
        # Returning data
        X = torch.stack(X).to(self.device)
        Y = torch.stack(Y).to(self.device)
        return X, Y
    

def train(model, data_loader, save_addr, num_epoch = 50, learning_rate = 1e-4, overwrite = False):
    # Creating save folder
    utils.create_dir(save_addr)
    model_addr = os.path.join(save_addr, 'model')
    utils.create_dir(model_addr)
    loss_addr = os.path.join(save_addr, 'loss')
    utils.create_dir(loss_addr)

    # Parameters for training
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    start_time = time.time()
    
    for epoch in range(num_epoch):
        batch_ct = 0
        epoch_loss = 0

        # Loading previous model
        epoch_addr = os.path.join(model_addr, f'{epoch}.pt')
        epoch_loss_addr = os.path.join(loss_addr, f'{epoch}.pt')

        if not overwrite and os.path.exists(epoch_addr) and os.path.exists(epoch_loss_addr):
            model.load_state_dict(torch.load(epoch_addr))
            epoch_loss = torch.load(epoch_loss_addr)

            print(f"Epoch: {epoch} Loaded\t\tLoss: {epoch_loss}")
        else:
            # Training next epoch
            for x, y in data_loader:
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_ct += 1
                print(f"\tBatch: {batch_ct}\tLoss: {round(loss.item(), 6)}\tTotal Loss: {round(epoch_loss/batch_ct, 6)}\tTime: {time.time()-start_time}")

            # Saving model after each epoch
            torch.save(model.state_dict(), epoch_addr)
            torch.save(epoch_loss/batch_ct, epoch_loss_addr)

            print(f"Epoch: {epoch}\tLoss: {round(epoch_loss/batch_ct, 6)}\tTime: {time.time() - start_time}")

def set_seed(rand_seed):
    torch.manual_seed(rand_seed)
