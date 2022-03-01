import torch
from torch.utils.data import Dataset, DataLoader


class ToyRandomDataset(Dataset):
    """A dataset of random numbers with dimension (num_data, num_features)
        with range (0, 1)
    """
    def __init__(self, num_data, num_features):
        self.num_data = num_data
        self.num_features = num_features
        self.data = torch.rand(self.num_data, self.num_features)

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    batch_size = 8
    num_data = 500
    num_features = 2
    dataset = ToyRandomDataset(num_data=num_data, num_features=num_features)
    print(f'Length of dataset: {len(dataset)}')
    queue = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    data = next(iter(queue))
    print(f'Data: {data}')
