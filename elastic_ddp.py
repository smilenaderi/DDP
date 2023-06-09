import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


NUM_REPLICAS = 8
BATCH_SIZE = 5000



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
        self.data = torch.randn((200000, 10000))  # Replace with your actual data
        self.targets = torch.randint(0, 10, (200000,))  # Replace with your actual targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        return sample, target
# import torch.utils.data.DistributedSampler as DistributedSampler
dataset = MyDataset()
# dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)


def demo_basic():
    start_time = time.time()
    # outputs = torch.randn((20000, 10000))  # Replace with your actual data
    # labels = torch.randint(0, 10, (20000,))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.\n")

    sampler = DistributedSampler(dataset,
                                 num_replicas=NUM_REPLICAS,
                                 rank=rank,
                                 shuffle=True,  # May be True
                                 seed=42)
    # Wrap train dataset into DataLoader
    dataloader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,  # Must be False!
                            num_workers=4,
                            sampler=sampler,
                            pin_memory=True)


    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    num_epochs = 10  # Number of epochs for training
    log_interval = 10  # Print the progress every 10 batches
    print(f' Loading Time: {time.time() - start_time} rank {rank} ')
    for epoch in range(num_epochs):
        dataloader.sampler.set_epoch(epoch)
        # if epoch == 1:
        # Training loop with QPS calculation

        query_count = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device_id), targets.to(device_id)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if epoch > 0:
            query_count += inputs.size(0)

            # if batch_idx % log_interval == 0 and epoch > 0:
            elapsed_time = time.time() - start_time
            qps = query_count / elapsed_time
            print("EPOCH:{}   Step [{}/{}], Loss: {:.4f}, QPS: {:.2f},  on device: {}, query_count: {}, time: {}"
                  .format(epoch, batch_idx + 1, len(dataloader), loss.item(), qps, device_id, query_count, time.time()-start_time))
    elapsed_time = time.time() - start_time

    print("elapsed_time", elapsed_time)
    total_qps = query_count / elapsed_time
    print("Total QPS: {:.2f}".format(total_qps))
    print(query_count)


if __name__ == "__main__":
    demo_basic()