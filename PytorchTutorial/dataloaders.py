import torch

from torch.utils.data import Dataset


X_train = torch.tensor([
[-1.2, 3.1],
[-0.9, 2.9],
[-0.5, 2.6],
[2.3, -1.1],
[2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])
X_test = torch.tensor([
[-0.8, 2.8],
[2.6, -1.6],
])
y_test = torch.tensor([0, 1])

class ToyDataset(Dataset):
    # in init function we used to to set up attributes that we can access later in the__getitem__ and __len__ methods
    def __init__(self, X, y):
        self.features = X
        self.labels = y
    
    #in getitem funciton we define instructions for returning exactly one item from the dataset via an index
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
    #the __len__ method contains instructions for retrieving the length of the dataset.
    def __len__(self):
        return self.labels.shape[0]

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

print(len(train_ds))


# from the dataloader class 
from torch.utils.data import DataLoader
torch.manual_seed(123)
train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,#When num_workers is set to 0, the data loading will be done in the main process and not in separate worker processes
    #In practice, having a substantially smaller batch as the last batch in a training epoch can disturb the convergence during training. To prevent this, set drop_last=True, which will drop the last batch in each epoch
    drop_last= True # to eliminate the last batch 
)
test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)