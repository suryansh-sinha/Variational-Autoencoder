import torch
from torch import nn
import torch.nn.functional as F


# Input Img -> Hidden dim -> mean, std -> reparameterization trick -> Decoder -> Output Img
class VariationalAutoencoder(nn.Module):
    
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()
        # For the encoder
        self.img_2hid = nn.Linear(input_dim, h_dim) # Converting img to hidden dim.
        self.hidden_2mu = nn.Linear(h_dim, z_dim)
        self.hidden_2sigma = nn.Linear(h_dim, z_dim)
        
        # For the decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)
        
        self.relu = nn.ReLU()
        
    def encode(self, x):
        # q_phi(z|x)
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hidden_2mu(h), self.hidden_2sigma(h)   # No ReLU because these can be negative values too.
        return mu, sigma
    
    def decode(self, z):
        # p_theta(x|z)
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))  # Want this to be b/w 0-1 as pixel values.
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_new = mu + sigma * epsilon    # Reparametrization trick.
        x_reconstructed = self.decode(z_new)
        return x_reconstructed, mu, sigma   # mu, sigma required for KL divergence part of loss
        
    
    
if __name__ == '__main__':
    x = torch.rand(4, 28*28)  # 28 * 28 = 784 [MNIST]
    vae = VariationalAutoencoder(28*28)
    x_reconstructed, mu, sigma = vae(x)
    print(mu.shape)
    print(sigma.shape)
    print(x_reconstructed.shape)