import torch
import torch.nn.functional as F
from datasets import load_dataset
from accelerate import Accelerator


import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc1 = nn.Linear(10000, 10000)
        self.fc2 = nn.Linear(10000, 10000)
        self.fc3 = nn.Linear(10000, 20000)
        self.fc4 = nn.Linear(20000, 10)


    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        # Create sample data
        self.data = torch.randn((20000, 10000))  # Replace with your actual data
        self.targets = torch.randint(0, 10, (20000,))  # Replace with your actual targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        return sample, target

dataset = MyDataset()

accelerator = Accelerator()

device = accelerator.device

model = ToyModel().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)

# dataset = load_dataset('my_dataset')
# data = torch.utils.data.DataLoader(dataset, shuffle=True)
dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

model, optimizer, data = accelerator.prepare(model, optimizer, dataloader)

model.train()
num_epochs = 10  # Number of epochs for training
log_interval = 10
for epoch in range(num_epochs):

    if epoch == 1:
        # Training loop with QPS calculation
        start_time = time.time()
        query_count = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):

          source = inputs.to(device)
          targets = targets.to(device)

          optimizer.zero_grad()

          output = model(source)
          loss = F.CrossEntropyLoss(output, targets)

          accelerator.backward(loss)

          optimizer.step()
          if epoch > 0:
              query_count += inputs.size(0)

          if batch_idx % log_interval == 0 and epoch > 0:
              elapsed_time = time.time() - start_time
              qps = query_count / elapsed_time
              print("EPOCH:{}   Step [{}/{}], Loss: {:.4f}, QPS: {:.2f}  on device: {}"
                    .format(epoch, batch_idx + 1, len(dataloader), loss.item(), qps, device_id))
elapsed_time = time.time() - start_time

print("elapsed_time", elapsed_time)
total_qps = query_count / elapsed_time
print("Total QPS: {:.2f}".format(total_qps))
print(query_count)