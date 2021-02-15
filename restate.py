import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
import os
import json
from sklearn.metrics import accuracy_score
import numpy as np


# self.linear = nn.Linear(dim * max_seq_length, 1)
# return self.linear(x.view(x.shape[0], -1))[:, 0]



class SingleLinear(nn.Module):
    def __init__(self, dim):
        super(SingleLinear, self).__init__()
        self.linear = nn.Linear(dim, 1, bias=True)

    def forward(self, x):
        return F.sigmoid(self.linear(x[:, 0])[:, 0])


class Restater(pl.LightningModule):
    def __init__(self, backend):
        super(Restater, self).__init__()
        self.backend = backend
        self.loss = nn.BCELoss()
        self.batch_size = batch_size

    def forward(self, x):
        return self.backend(x)

    def training_step(self, batch, _):
        x, y = batch
        y_ = self.backend(x)
        loss = -self.loss(y_.float(), y.float())
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        return DataLoader(StateDataset(model_name, "train"), batch_size=self.batch_size, shuffle=True)


class StateDataset(Dataset):
    def __init__(self, directory, split):
        self.split = split
        self.dir = f"states/{directory}/{split}"
        self.items = os.listdir(self.dir)
        self.labels = [float(line.get("label", 0)) for line in map(json.loads, open(f"datasets/DaNetQA/{split}.jsonl"))]

    def __getitem__(self, index):
        x = torch.load(self.dir + '/' + self.items[index])
        x = torch.cat((x, torch.zeros((max_seq_length - x.shape[0], x.shape[1]))), dim=0)
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.items)


model_name = "platinum/xlarge"
model_save = "zerode/xlarge"
n_epochs = 10
batch_size = 64
max_seq_length = 512


def save(model, data):
    split = data.split
    ys = []
    ys_ = []
    with torch.no_grad():
        for x, y in tqdm(DataLoader(data, shuffle=False)):
            ys += [z.item() for z in y]
            lg = F.sigmoid(model(x))
            ys_ += [z.item() for z in lg]
    scores = []
    for thresh in range(100):
        for flip in [False, True]:
            y = np.array(ys_)
            if flip:
                y = 1 - y
            scores.append((accuracy_score(np.array(ys), y > thresh / 100), thresh))
    print('Accuracy:', min(scores))
    open(f"scores/{model_save}.{split}.scores", 'w').write('\n'.join(map(str, ys)))


if __name__ == '__main__':
    print("Loading data...")
    train = StateDataset(model_name, "train")
    val = StateDataset(model_name, "val")
    test = StateDataset(model_name, "test")

    print("Loading model...")
    model = Restater(SingleLinear(train[0][0].shape[-1]))
    save(model, val)

    print("Training...")
    trainer = pl.Trainer(min_epochs=n_epochs, max_epochs=n_epochs, check_val_every_n_epoch=1000, gpus=1)
    trainer.fit(model)

    for data in (val, test):
        print("Writing", data.split)
        save(model, data)
