import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time

# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10000, 10000)
        self.fc2 = nn.Linear(10000, 10000)
        self.fc3 = nn.Linear(10000, 20000)
        self.fc4 = nn.Linear(20000, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

# Initialize your model
model = MyModel()

# Define your dataset and dataloader
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
dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

# Check if a GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

# Define your optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Define other necessary variables
num_epochs = 10  # Number of epochs for training
log_interval = 10  # Print the progress every 10 batches




for epoch in range(num_epochs):
    if epoch == 1:
        # Training loop with QPS calculation
        start_time = time.time()
        query_count = 0
    print("Epoch [{}/{}]".format(epoch+1, num_epochs))
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch > 0:
            query_count += inputs.size(0)

        if batch_idx % log_interval == 0 and epoch > 0 :
            elapsed_time = time.time() - start_time
            qps = query_count / elapsed_time
            print("  Step [{}/{}], Loss: {:.4f}, QPS: {:.2f}"
                  .format(batch_idx+1, len(dataloader), loss.item(), qps))

# Calculate total QPS
elapsed_time = time.time() - start_time

print("elapsed_time" , elapsed_time)
total_qps = query_count / elapsed_time
print("Total QPS: {:.2f}".format(total_qps))
print(query_count)
