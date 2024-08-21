from tqdm import tqdm
import os, sys
import os.path as osp
import time
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
pd.options.mode.copy_on_write = True

# import sklearn as skl
# from sklearn import tree
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import DataLoader


def process_data(data, use_luxery=True):
    headers = [c for c in data.columns]
    print(headers)
    # print(data.Transported.value_counts())
    if not use_luxery:
        seldata = data[['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP']]
    else:
        seldata = data[['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
        seldata['RoomService'] = seldata['RoomService'].fillna(0)
        seldata['FoodCourt'] = seldata['FoodCourt'].fillna(0)
        seldata['ShoppingMall'] = seldata['ShoppingMall'].fillna(0)
        seldata['Spa'] = seldata['Spa'].fillna(0)
        seldata['VRDeck'] = seldata['VRDeck'].fillna(0)
    seldata['HomePlanet'] = seldata['HomePlanet'].fillna('na')
    seldata['CryoSleep'] = seldata['CryoSleep'].fillna(False)
    seldata['Cabin'] = seldata['Cabin'].fillna('n/n/n')
    seldata['Destination'] = seldata['Destination'].fillna('na')
    seldata['Age'] = seldata['Age'].fillna(-1)
    seldata['VIP'] = seldata['VIP'].fillna(False)
    cabin = [c if not pd.isnull(c) else 'n/n/n' for c in data['Cabin']]
    for cab in cabin:
        try:
            cab.split('/')
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()
    cabin = [c.split('/') for c in cabin]
    cabindeck = np.array([c[0] for c in cabin])
    cabinnum = np.array([float(c[1]) if c[1] != 'n' else 0 for c in cabin])
    cabinside = np.array([c[2] for c in cabin])

    def categorize(arr):
        if isinstance(arr, pd.core.series.Series):
            arr = arr.to_numpy()
        try:
            categories = np.unique(arr)
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()
        print(categories)
        arr = arr.copy()
        for ci, cat in enumerate(categories):
            arr[arr == cat] = ci
        try:
            return arr.astype(float)
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()

    seldata['CabinDeck'] = categorize(cabindeck)
    seldata['CabinNum'] = cabinnum.astype(float)
    seldata['CabinSide'] = categorize(cabinside)

    # if pd.isnull(seldata['HomePlanet']).any():
    #     seldata['HomePlanet'] = 
    seldata['HomePlanet'] = categorize(seldata['HomePlanet'].to_numpy())
    seldata['Destination'] = categorize(seldata['Destination'].to_numpy())

    if not use_luxery:
        seldata = seldata[['HomePlanet', 'CryoSleep', 'CabinDeck', 'CabinNum', 'CabinSide', 'Destination', 'Age', 'VIP']]
    else:
        seldata = seldata[['HomePlanet', 'CryoSleep', 'CabinDeck', 'CabinNum', 'CabinSide', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]

    # seldata['Transported'] = data['Transported']

    print(seldata.head(3))

    return seldata


class Mish(nn.Module):
    """
    Mish activation function from:
    "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
    https://arxiv.org/abs/1908.08681v1
    x * tanh( softplus( x ) )
    """

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh( F.softplus(x) )


class Sine(nn.Module):
    """
    Sine activation function from:
    "Sine: Implicit Neural Representations with Periodic Activation Functions"
    https://www.vincentsitzmann.com/siren/
    sine(x)
    """

    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, x):
        return torch.sin(x)


activations = {'relu': nn.ReLU, 'mish': Mish, 'selu': nn.SELU, 'elu': nn.ELU, 'sine': Sine}


class FCNet(nn.Module):
    """Fully Connected Network"""

    def __init__(self, infeats=10, outfeats=10, layers=[15, 10, 15], activation='relu',
        lastivation=None, batchnorm=False, layernorm=False, dropout=0
    ):
        super(FCNet, self).__init__()
        self.infeats = infeats
        self.outfeats = outfeats
        self.layers = layers
        if isinstance(activation, str):
            activation = activations[activation]
        self.activation = activation
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.dropout = dropout

        model = []
        layers = [infeats] + layers
        for ix, layer_size in enumerate(layers[1:]):
            model.append(nn.Linear(layers[ix], layer_size))
            if layernorm:
                model.append(nn.LayerNorm(layer_size))
            model.append(self.activation())
            if batchnorm and ix < (len(layers)-1):
                model.append(nn.BatchNorm1d(layer_size))
            if dropout > 0 and ix < (len(layers)-1):
                model.append(nn.Dropout(dropout))
        model.append(nn.Linear(layers[-1], outfeats))
        self.model = nn.Sequential( *model )
        self.lastivation = lastivation

    def forward(self, xdata):
        ydata = self.model(xdata)
        if self.lastivation is not None:
            ydata = self.lastivation(ydata)
        return ydata
    
    def loss_fn(self, xdata, target):
        pred = self(xdata)
        return F.binary_cross_entropy_with_logits(pred, target)

    def evaluate(self, dataset):
        self.eval()
        loss = 0
        acc = 0
        for d, t in dataset:
            d = d.to(self.get_device())
            t = t.to(self.get_device())
            pred = F.sigmoid(self(d)) > 0.5
            acc += (pred == t).sum().item() / len(d)
            loss += self.loss_fn(d, t).item()
        return loss / len(dataset), acc / len(dataset)
    
    def get_device(self):
        return next(self.parameters()).device


def tensor_process(data):
    halfmax_cabindeck = 4
    halfmax_cabinnum = 1000
    halfmax_age = 60
    halfmax_lux = 250
    # data = torch.tensor(data[['HomePlanet', 'CryoSleep', 'CabinDeck', 'CabinNum', 'CabinSide', 'Destination', 'Age', 'VIP']].to_numpy().astype(float), dtype=torch.float32)
    data = torch.tensor(data.to_numpy().astype(float), dtype=torch.float32)
    # normalize CabinDeck
    data[:, 2] = data[:, 2] / halfmax_cabindeck - 1
    # normalize CabinNum
    data[:, 3] = data[:, 3] / halfmax_cabinnum - 1
    # normalize age
    data[:, 6] = data[:, 6] / halfmax_age - 1
    if data.shape[1] > 8:
        # normalize luxeries
        data[:, 8] = data[:, 8] / halfmax_lux - 1
        data[:, 9] = data[:, 9] / halfmax_lux - 1
        data[:, 10] = data[:, 10] / halfmax_lux - 1
        data[:, 11] = data[:, 11] / halfmax_lux - 1
        data[:, 12] = data[:, 12] / halfmax_lux - 1
    return data


def main():
    batch_size = 100
    use_luxery = True
    # ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name', 'Transported']
    data = pd.read_csv("data/train.csv") 
    tdata = process_data(data, use_luxery=use_luxery)
    validation_data = pd.read_csv("data/valid.csv")
    vdata = process_data(validation_data, use_luxery=use_luxery)

    tdata = tensor_process(tdata)
    ttargets = torch.tensor(data['Transported'], dtype=torch.float32).reshape(-1, 1)
    trainset = torch.utils.data.TensorDataset(tdata, ttargets)
    trainload = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    vdata = tensor_process(vdata)
    vtargets = torch.tensor(validation_data['Transported'], dtype=torch.float32).reshape(-1, 1)
    validset = torch.utils.data.TensorDataset(vdata, vtargets)
    validload = DataLoader(
        validset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    datasamp, targsamp = trainset[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_grad = None

    model_name = '50bnlndrmish'

    model = FCNet(len(datasamp), 1, layers=[50, 50, 50, 50, 50], activation='mish', batchnorm=True, layernorm=True, dropout=0.5)
    model.to(device)
    print(model)

    optim = torch.optim.Adam(
        model.parameters(),
        lr=2e-4,
        weight_decay=5e-4
    )
    clip_grad = lambda x : nn.utils.clip_grad_norm_(
        x, 1
    )

    model.train()
    start_epoch = 0
    end_epoch = 1000
    eval_batch = 25
    save_best_after = 100
    save_batch = 25
    train_losses = []
    train_acces = []
    valid_losses = []
    valid_acces = []

    def save_model(epoch):
        checkpoint = {}
        checkpoint['model'] = model.state_dict()
        checkpoint['optim'] = optim.state_dict()
        checkpoint['train_losses'] = train_losses
        checkpoint['valid_losses'] = valid_losses
        checkpoint['valid_acces'] = valid_acces
        checkpoint['epoch'] = epoch
        filename = 'results/deepnet/deepnet_e{}_{}.torch'.format(epoch, model_name)
        torch.save(checkpoint, filename)
        plt.plot(train_losses)
        plt.plot(valid_losses)
        plt.plot(valid_acces)
        plt.savefig('results/deepnet/deepnet_epoch{}_{}.png'.format(epoch, model_name))
        plt.clf()
        print('saved to', filename)

    for epoch in range(start_epoch, end_epoch + 1):
        loss = 0.
        mean_loss = 0.
        mean_acc = 0.
        pbar = tqdm(total=len(trainload), file=sys.stdout)
        desc = f'Train Epoch {epoch}'
        pbar.set_description(desc)
        for iter, (batch, target) in enumerate(trainload):
            batch = batch.to(device)
            target = target.to(device)
            loss = model.loss_fn(batch, target)
            optim.zero_grad()
            loss.backward()
            if clip_grad is not None:
                clip_grad(model.parameters())
            optim.step()
            mean_loss += loss.item()

            ploss = mean_loss / (iter + 1)
            desc = f'Train Epoch {epoch}: loss={ploss:.3f}'
            pbar.set_description(desc)
            pbar.update(1)
        pbar.close()
        train_losses.append(mean_loss / len(trainload))

        if epoch % eval_batch == 0:
            eval_time = time.time()
            vloss, vacc = model.evaluate(validload)
            valid_losses.append(vloss)
            valid_acces.append(vacc)
            print('eval time:', time.time() - eval_time)
            print('eval acc:', vacc)
            model.train()
        else:
            if len(valid_losses) > 0:
                valid_losses.append(valid_losses[-1])
                valid_acces.append(valid_acces[-1])
            else:
                valid_losses.append(train_losses[-1])
                valid_acces.append(0)

        if epoch >= save_best_after and epoch % save_batch == 0:
            save_model(epoch)

    best_epoch = np.argmax(valid_acces)
    print('best epoch:', best_epoch, valid_acces[best_epoch])
    filename = 'results/deepnet/deepnet_e{}_{}.torch'.format(best_epoch, model_name)
    model_info = torch.load(filename)
    torch.save(model_info, 'results/deepnet/best_model_{}.torch'.format(model_name))


if __name__ == '__main__':
    main()


