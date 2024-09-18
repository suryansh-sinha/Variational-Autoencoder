import torch
from torch import nn, optim
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from model import VariationalAutoencoder
from tqdm import tqdm

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
INPUT_DIM = 28*28   # 784
H_DIM = 200 # Adds more compute
Z_DIM = 32  # Constitutes for compression
NUM_EPOCHS = 30
BATCH_SIZE = 128
LR_RATE = 3e-4  # Karpathy constant

# Dataset loading
dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VariationalAutoencoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)

# Reconstruction Loss - 
loss_fn = nn.BCELoss(reduction='sum')   # y is pixel values of the image

# Starting training
train_loss = []
for epoch in tqdm(range(NUM_EPOCHS)):
    loop = tqdm(enumerate(train_loader))
    for i, (x, _) in loop:
        # Forward pass
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)    # Flattening [batch_size, 28*28]
        x_reconstructed, mu, sigma = model(x)
        
        # Compute loss
        reconstruction_loss = loss_fn(x_reconstructed, x)
        # In the paper, we want to maximize this, but in torch we need to minimize the loss.
        # So, we are using the -kl_divergence here.
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        
        # Backprop
        loss = reconstruction_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
        

def inference(digit, num_examples=1):
    """
    Generate (num_examples) of a particular digit.
    Specifically, we extract an example of each digit,
    then after we have the mu, sigma representation for
    each digit, we can sample from that.
    
    After we sample, we can run the decoder part of VAE and generate examples.
    
    digit: The digit you want to generate.
    num_examples: The number of samples you want to generate for that image.
    """
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x.to(DEVICE))
            idx += 1
        if idx == 10:
            break
        
    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))
        
    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f'generated/generated_{digit}_ex{example}.png')

for idx in range(10):
    inference(idx, num_examples=5)