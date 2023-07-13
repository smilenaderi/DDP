# !torchrun --nnodes=1 --nproc_per_node=2  --rdzv_endpoint=127.0.0.1:29409 elastic_ddp.py

import time
from lib310_lite import Laser
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

NUM_REPLICAS = 2
BATCH_SIZE = 100


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc1 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc4(x)
        return x


class DataAcess():

    def __init__(self, device_id):
        self.db_conf = {
            'host': '18.246.45.68',
            'user': 'citizix',
            'password': 'S3cret@310Vz2',
            'database': 'laser'
        }
        self.laser = Laser(self.db_conf)
        self.q = 'SELECT row_id FROM fs_indexable WHERE len <= 300 and token_size >= 10 AND stage = 1 ORDER BY RAND(42);'
        self.pool = self.laser.get_pool(self.q)
        self.i_index = len(self.pool) // NUM_REPLICAS * device_id
        self.laser.start_buffering(sample_number=BATCH_SIZE, starting_batch_number=self.i_index)
        print(f'DATA access loaded rank {device_id} index {self.i_index}')

    def len(self):
        return len(self.pool)

    def getitem(self):
        return self.laser.get_batch()


def demo_basic():
    start_time = time.time()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()

    dataAccess = DataAcess(device_id)
    print(f"Start running basic DDP example on rank {rank}.\n")

    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    num_epochs = 10  # Number of epochs for training
    print(f' Loading Time: {time.time() - start_time} rank {rank} ')
    for epoch in range(num_epochs):

        query_count = 0
        for i in range(dataAccess.len() // BATCH_SIZE):
            inputs = dataAccess.getitem()
            inputs = np.vstack(inputs['id']).astype(float)
            inputs = torch.from_numpy(inputs).float().to(device_id)

            query_count += inputs.size(0)
            if (i % 500 == 0):
                print(f'QPS on rank {rank}  : {query_count / (time.time() - start_time)}')



if __name__ == "__main__":
    demo_basic()


    # !torchrun --nnodes=1 --nproc_per_node=2  --rdzv_endpoint=127.0.0.1:29409 elastic_ddp.py