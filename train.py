import torch
import torch.nn as nn
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import SimpleMNISTNN

def get_mnist_dataloaders(batch_size, test_batch_size=1, shuffle_test=False):
    train_dl = DataLoader(MNIST('.', transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])), batch_size = batch_size)
    test_dl = DataLoader(MNIST('.', train=False, transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])), batch_size = test_batch_size, shuffle=shuffle_test)
    return train_dl, test_dl

def train_mnist(model_class, epochs=1, batch_size=16, test_net=False, **model_params):
    """
    This is how we get our "training networks"
    """
    net = model_class(**model_params)
    train, test = get_mnist_dataloaders(batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    losses = []
    for _ in range(epochs):
        for data, label in tqdm(train):
            optimizer.zero_grad()
            out = net(data)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                losses.append(loss.numpy())
    
    if test_net:
        accuracy = test_network(net, test)
        return net, losses, accuracy
    return net, losses

def test_network(net, test, return_outs=False):
    correct = 0
    total = 0.0
    outs = []
    for data, label in test:
        total += 1.0
        prob, out = torch.max(net(data), dim=-1)

        correct += (out == label)
    return correct / total


def flat_to_net(flat_net):
    test_net = SimpleMNISTCNN()
    index = 0
    for p in test_net.parameters():
        end_index = torch.cumprod(torch.tensor(p.shape), 0)[-1] + index
        p = flat_net[index:end_index].reshape(p.shape)
        index = end_index

class MNISTNetDataset(torch.utils.data.Dataset):
    """
    To make batching easy
    """
    def __init__(self, nets, batch_size):
        self.nets = nets
        self.batch_size = batch_size

    def __getitem__(self, idx):
        return self.nets[idx].get_flat_network()

    def __len__(self):
        return len(self.nets)

def generate_meta_dataloader(nets, batch_size=64, mnist_batch_size=256):
    dset = MNISTNetDataset(nets, batch_size)
    dl = DataLoader(dset, shuffle=False, batch_size=batch_size)
    return dl


def train_vae(vae, dl, batch_size=64, mnist_batch_size=256):
    """
    Takes in model and data loader and runs 1 epoch of training, using
    Symmetric KL Divergence between the output net and the input net and the 
    Euclidean Distance between weights as a reconstruction loss, and VAE KL loss
    to regularize latent space

    NOTE: This is an old version and the new, not clean one is in train_vae.ipynb
    DO NOT USE
    """
    optimizer = torch.optim.Adam(vae.parameters())
    test = DataLoader(MNIST('.', train=False, transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])), batch_size = batch_size, shuffle=False)
    losses = []
    for net, outs in dl:
        mean, std, output = vae.forward_train(flat_net)
        new_nets = [flat_to_net(flat) for flat in output]
        symmetric_kl = 0
        total = 0
        for n in new_nets:
            for out, (batch, _) in zip(outs, test):
                new_out = n(batch)
                total += 1.0
                # symmetrized KL
                symmetric_kl += (0.5 * torch.log(out / new_out) * out + 0.5 * torch.log(new_out / out) * new_out) / batch_size
        symmetric_kl /= total
        vae_loss = symmetric_kl + vae.calc_loss(mean, std)
        vae_loss.backward()
        losses.append(vae_loss.detach())
        optimizer.step()
        # print(vae_loss)
    return losses
                





    