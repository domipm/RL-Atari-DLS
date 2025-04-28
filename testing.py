# Implementation of Vector-Quantized Variational Auto-Encoder (VAE)
# This version of the script trains the model on previously-generated sample of images (train split)
# and evaluates it on similar images (test split)

# Inputs:
#   sequence of frames (*.png) located in ./frames/(Game-Name)/
# Outputs:
#   ...

import  torchsummary
import  torch.nn                    as nn
import  torch.optim                 as optim
import  torch.nn.functional         as F

import  os
import  torch

from    PIL                     import Image

from    torchvision.transforms  import v2
from    torch.utils.data        import Dataset

import  matplotlib.pyplot   as plt

import  torchvision

from    torch.utils.data    import DataLoader



'''CUSTOM FRAME DATASET LOADER'''



# Define default transform to apply to images
transform_default = v2.Compose([
    # Resize all images
    v2.Resize((128, 64)),
    # Convert to grayscale
    # transforms.Grayscale(),
    # Convert to tensor object
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    # Normalize image
    # transforms.Normalize((0.5,), (0.5,))
])

# Custom class for loading frames
class FramesDataset(Dataset):

    # Initialization function for dataset
    def __init__(self, directory, transform = transform_default):

        # Initialize parent class
        super().__init__()

        # Initialize directory
        self.directory = directory
        # List all images within directory

        self.images = sorted( [f for f in os.listdir(directory) if not f.startswith('.')] )
        # Set transform to use on images
        self.transform = transform

        return
    
    # Return size of dataset (numer of all images)
    def __len__(self):

        return len(self.images)
    
    # Get single item from dataset (for indexing dataset[i] returns i-th sample)
    def __getitem__(self, index):

        # Get the image with specificed index
        image = self.images[index]
        # Get the path to that image (directory/class_plural/class.*)
        image_path = os.path.join(self.directory, image)
        # Open image using PIL.Image and apply relevant transformations
        image = Image.open(image_path)
        # Apply transformations if given
        if self.transform != None: image = self.transform(image)
        # Otherwise, just transform to tensor
        else: image = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float, scale=True)])(image)
        # Return tensor image
        return image



'''VQ-VAE MODEL ARCHITECTURE DEFINITION'''



class VQ_VAE(nn.Module):

    # Initialization function with definitions for all layers
    def __init__(self, embedding_num = 512, embedding_dim = 64, l_codebook = 1, l_commit = 1, in_shape = torch.Size([0, 0, 0, 0])):

        # Initialize parent class
        super().__init__()

        # Input image shape
        self.in_shape = in_shape

        # Embedding number K (Number of vectors in codebook) 
        self.embedding_num = embedding_num
        # Embedding dimension D (Size of vectors in codebook)
        self.embedding_dim = embedding_dim
        # Embeddings (Codebook) shape (K, D)
        self.codebook = nn.Embedding(self.embedding_num, self.embedding_dim)
        # Initialize codebook weights uniformly
        self.codebook.weight.data.uniform_(-1/self.embedding_num, 1/self.embedding_num)

        # Encoder
        self.encoder = nn.Sequential(

            # Convolutional layer (B, C = 3, H, W) -> (B, C = 16, H/2, W/2)
            nn.Conv2d(in_channels = in_shape[1], 
                      out_channels = 16, 
                      kernel_size = 4, stride = 2, padding = 1),
            # Batch normalization layer
            nn.BatchNorm2d(num_features = 16),
            # ReLU activation function
            nn.ReLU(),

            # Convolutional layer (B, C = 3, H, W) -> (B, C = 32, H/2, W/2)
            nn.Conv2d(in_channels = 16, 
                      out_channels = 32, 
                      kernel_size = 4, stride = 2, padding = 1),
            # Batch normalization layer
            nn.BatchNorm2d(num_features = 32),
            # ReLU activation function
            nn.ReLU(),

            # Convolutional layer (B, C = 3, H, W) -> (B, C = 16, H/2, W/2)
            nn.Conv2d(in_channels = 32, 
                      out_channels = self.embedding_dim, 
                      kernel_size = 4, stride = 2, padding = 1),
            # Batch normalization layer
            nn.BatchNorm2d(num_features = self.embedding_dim),
            # ReLU activation function
            nn.ReLU(),

        )

        # Decoder
        self.decoder = nn.Sequential(

            # Transposed Convolutional layer (B, C = 16, H, W)
            nn.ConvTranspose2d(in_channels = 64, 
                      out_channels = 32, 
                      kernel_size = 4, stride = 2, padding = 1),
            # Batch normalization layer
            nn.BatchNorm2d(num_features = 32),
            # ReLU activation function
            nn.ReLU(),

            # Convolutional layer (B, C = 3, H, W) -> (B, C = 32, H/2, W/2)
            nn.ConvTranspose2d(in_channels = 32, 
                      out_channels = 16, 
                      kernel_size = 4, stride = 2, padding = 1),
            # Batch normalization layer
            nn.BatchNorm2d(num_features = 16),
            # ReLU activation function
            nn.ReLU(),

            # Convolutional layer (B, C = 3, H, W) -> (B, C = 16, H/2, W/2)
            nn.ConvTranspose2d(in_channels = 16, 
                      out_channels = self.in_shape[1], 
                      kernel_size = 4, stride = 2, padding = 1),
            # Batch normalization layer
            nn.BatchNorm2d(num_features = self.in_shape[1]),
            # ReLU activation function
            nn.Tanh(),

        )

        return
    
    # Forward pass of the networks, return output from last dense layer 
    def forward(self, x):

        # Input image x: (B, C = 3, H, W)
        print("x.shape = ", x.shape)

        '''ENCODER'''

        # Encode image: (B, C = 3, H, W) -> (B, D, H', W')
        z_e = self.encoder(x)
        print("z_e.shape = ", z_e.shape)

        '''VECTOR QUANTIZATION'''

        # Perform vector quantization

        # Permute tensor to shape (B, H, W, D) creating new copy (contiguous)
        z = z_e.permute(0, 2, 3, 1).contiguous()
        print("z.shape = ", z.shape)

        # Flatten batch and spatial dimensions together
        z = z.view(-1, self.embedding_dim)
        print("z.shape = ", z.shape)

        # Compute distances between each vector and codebook vectors
        dist = (
            torch.sum(z**2, dim=1, keepdim=True)  # (2048, 1)
            + torch.sum(self.codebook.weight**2, dim=1)     # (512,)
            - 2 * torch.matmul(z, self.codebook.weight.t())  # (2048, 512)
        )

        # Find nearest neighbor for each feature vector
        encoding_indices = torch.argmin(dist, dim = 1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.embedding_num)
        encodings.scatter_(1, encoding_indices, 1)

        # Lookup embeddings
        quantized = torch.matmul(encodings, self.codebook.weight).view(z_e.permute(0,2,3,1).contiguous().shape)

        # Loss terms
        e_latent_loss = F.mse_loss(quantized.detach(), z_e.permute(0,2,3,1).contiguous())
        q_latent_loss = F.mse_loss(quantized, z_e.permute(0,2,3,1).contiguous().detach())
        loss = q_latent_loss + e_latent_loss

        quantized = z_e.permute(0,2,3,1).contiguous() + (quantized - z_e.permute(0,2,3,1).contiguous()).detach()

        # Permute to shape (B, D, H, W)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        print("z_q.shape = ", quantized.shape)

        '''DECODER'''

        z_d = self.decoder(quantized)
        print("z_d.shape = ", z_d.shape)

        return z_d, loss
    



'''VQ_VAE TRAINING'''



# Directory of images
path_img = "./frames/Testing-Cats/train/"

# Load dataset using custom dataloader
dataset_train = FramesDataset(path_img)
# Load images from dataset batch-wise using dataloader
dataloader_train = DataLoader(dataset_train, batch_size = 16, shuffle = True)

# Initialize VQ-VAE model
model = VQ_VAE(
    embedding_num=512,
    embedding_dim=64,
    l_codebook=1,
    l_commit=1,
    in_shape=torch.Size([16, 3, 128, 64])
)

# Training parameters
epochs = 20
optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.MSELoss()

for epoch in range(epochs):

    print("Epoch ", epoch)

    for image in dataloader_train:

        im = image.float()

        optimizer.zero_grad()

        out, loss = model(im)

        recon_loss = criterion(out, im)
        loss = recon_loss + loss

        print("loss = ", loss.item())

        loss.backward()
        optimizer.step()



'''RECONSTRUCTION'''

# Directory of images
path_img = "./frames/Testing-Cats/test/"

# Load dataset using custom dataloader
dataset_test = FramesDataset(path_img)
# Load images from dataset batch-wise using dataloader
dataloader_test = DataLoader(dataset_test, batch_size = 16, shuffle = False)


# Show sample batch of images after transformation to tensor objects
images = next(iter(dataloader_test))
grid_img = torchvision.utils.make_grid(images, nrow=4)
# Convert tensor object to numpy array
npimg = grid_img.permute(1, 2, 0).numpy()
# Plot setup and show image
plt.figure(figsize=(8, 8))
plt.imshow(npimg)
plt.title("Test Images (Pre-Processed)")
plt.axis('off')
plt.savefig("testing_0.png", dpi=300)

# Generate reconstructed image
model.eval()

reconstructed, _ = model(images)

grid_img = torchvision.utils.make_grid(reconstructed.detach(), nrow=4)
# Convert tensor object to numpy array
npimg = grid_img.permute(1, 2, 0).numpy()
# Plot setup and show image
plt.figure(figsize=(8, 8))
plt.imshow(npimg)
plt.title("Test Images (Reconstructed)")
plt.axis('off')
plt.savefig("testing_1.png", dpi=300)




