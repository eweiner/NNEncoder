import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class Flatten(nn.Module):
    """
    "Layer" to transition from Convolutions to fully connected
    """
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        return x.view(x.shape[0], -1)
    
class Node:
    """
    Not used, potentially would for Graph Neural Network Encodings
    """
    def __init__(self, parameters):
        self.parameters = parameters
        self.children = []

class SimpleMNISTNN(nn.Module):
    """
    Multi-layer perceptron, we don't use because it has so many parameters...
    """
    def __init__(self):
        super(SimpleMNISTNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 30)
        self.fc2 = nn.Linear(30, 10)
        self.thingies = nn.ModuleList([self.fc1, self.fc2])
        self.softmax = nn.Softmax(dim=-1)
        self.flatten = Flatten()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(self.flatten(x.squeeze(1)))))
    
    def get_flat_network(self):
        net_params = []
        for p in self.parameters():
            net_params.append(p.view(1, -1))
            
        return torch.cat(net_params, 1).squeeze(0)

class SimpleMNISTCNN(nn.Module):
    """
    Current model we are embedding
    """
    def __init__(self):
        super(SimpleMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 5, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(245, 10)
        self.classifier_list = nn.ModuleList([self.conv1, 
                                    self.relu, 
                                    self.maxpool,
                                    self.conv2,
                                    self.relu,
                                    self.maxpool,
                                    Flatten(),
                                    self.fc])
        self.classifier = nn.Sequential(*self.classifier_list)
    
    def forward(self, x):
        return self.classifier(x)

    def get_flat_network(self):
        net_params = []
        for p in self.parameters():
            net_params.append(p.reshape(1, -1))
            
        return torch.cat(net_params, 1).squeeze(0)


class Embedder(nn.Module):
    """
    Embedding part of VAE
    """
    def __init__(self, input_size, hidden_size, hidden_layers, embedding_size):
        super(Embedder, self).__init__()
        self.embedder = nn.ModuleList([nn.Linear(input_size, hidden_size), nn.LeakyReLU()])
        for _ in range(hidden_layers):
            self.embedder.extend([nn.Linear(hidden_size, hidden_size), nn.LeakyReLU()])
        # self.embedder.append(nn.Linear(hidden_size, embedding_size))
        self.forward_embedder = nn.Sequential(*self.embedder)
        self.get_mean = nn.Linear(hidden_size, embedding_size)
        self.get_logvar = nn.Linear(hidden_size, embedding_size)

    def forward(self, x):
        hidden = self.forward_embedder(x)
        return self.get_mean(hidden), self.get_logvar(hidden)

class Decoder(nn.Module):
    """
    Decoder part of vae
    """
    def __init__(self, embedding_size, hidden_size, hidden_layers, output_size):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList([nn.Linear(embedding_size, hidden_size), nn.LeakyReLU()])
        for _ in range(hidden_layers):
            self.decoder.extend([nn.Linear(hidden_size, hidden_size), nn.LeakyReLU()])
        self.decoder.append(nn.Linear(hidden_size, output_size))
        self.forward_decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        return self.forward_decoder(x)

class VanillaVAE(nn.Module):
    """
    Vanilla VAE :)
    """
    def __init__(self, input_size, 
                    encoder_hidden_size,
                    encoder_num_hidden,
                    embedding_size,
                    decoder_hidden_size, 
                    decoder_num_hidden):
        super(VanillaVAE, self).__init__()
        self.encoder = Embedder(input_size, encoder_hidden_size, encoder_num_hidden, embedding_size)
        self.decoder = Decoder(embedding_size, decoder_hidden_size, decoder_num_hidden, input_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def reparametrize(self, mu, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.rand(std.size())
        return eps * std + mu

    def forward(self, x):
        _, _, output = self.forward_train(x)
        return output

    def forward_train(self, x):
        mean, logvar = self.encode(x)
        encodings = self.reparametrize(mean, logvar)
        output = self.decode(encodings)
        return mean, logvar, output

    def calc_loss(self, mean, logvar, constant=0.5):
        KLD_element = #mean.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5) / mean.shape[0] * constant
        return KLD
        

def test_train():
    """
    Made this to make sure we can get VAE to converge on fake, simple data
    """
    net = VanillaVAE(5, 20, 2, 2, 20, 2)
    test_input = torch.arange(20, dtype=torch.float).reshape(-1, 5) * 1.0 / 20.0
    optimizer = torch.optim.Adam(net.parameters())
    for i in range(1000):
        optimizer.zero_grad()
        mean, logvar, output = net(test_input)
        # print(mean)
        # print(logvar)
        # print(output)
        loss = net.calc_loss(test_input, mean, logvar, output)
        if i % 5 == 0:
            print(loss)
        loss.backward()
        optimizer.step()
    print("Expectation: ", torch.arange(5, dtype=torch.float).unsqueeze(0) / 20)
    print(net(torch.arange(5, dtype=torch.float).unsqueeze(0) / 20))
    # net(test_input)

if __name__ == "__main__":
    test_train()
    
        