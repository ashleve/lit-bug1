
from pytorch_lightning import LightningModule
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

import hydra


class RandomDataset(Dataset):
    def __init__(self, size, num_samples):
        self.len = num_samples
        self.data = torch.randn(num_samples, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def loss(self, batch, prediction):
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def training_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    def test_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        self.log('fake_test_acc', loss)
        return {"y": loss}

    def configure_optimizers(self):
        optim_conf = {
            "_target_": "torch.optim.Adam",
            "lr": 0.001,
            "eps": 1e-08,
            "weight_decay": 0,
            "betas": [ 0.9, 0.999 ]
        }
        
        # this doesnt work :(
        optim = hydra.utils.instantiate(optim_conf, params=self.parameters())
        
        # this works
        # optim = torch.optim.Adam(self.parameters())

        return optim


num_samples = 10000

train = RandomDataset(32, num_samples)
train = DataLoader(train, batch_size=32)

val = RandomDataset(32, num_samples)
val = DataLoader(val, batch_size=32)

test = RandomDataset(32, num_samples)
test = DataLoader(test, batch_size=32)

model = BoringModel()

trainer = pl.Trainer(
    max_epochs=1, 
    progress_bar_refresh_rate=20
)

trainer.fit(model, train, val)

trainer.test(test_dataloaders=test)
