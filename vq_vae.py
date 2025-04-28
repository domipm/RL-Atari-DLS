# Implementation of Vector-Quantized Variational Auto-Encoder (VAE)
# This version of the script trains the model on previously-generated sample of images (train split)
# and evaluates it on similar images (test split)

# Inputs:
#   sequence of frames (*.png) located in ./frames/(Game-Name)/
# Outputs:
#   ...

import  torchsummary
import  torch.nn as nn

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
    v2.Resize((128, )*2),
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
        self.images = sorted(os.listdir(directory))
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
    def __init__(self, embedding_num, embedding_dim, in_shape = None):

        # Initialize parent class
        super().__init__()

        # Print model summary if given input shape
        if in_shape != None:
            torchsummary.summary(self, in_shape)

        return
    
    # Forward pass of the networks, return output from last dense layer 
    def forward(self, x):

        # Pass 

        return



'''DATASET SAMPLE BATCH VISUALIZATION'''



# Directory of images
path_img = "./frames/Testing-Cats/test/"
# Directory for output
path_out = "./output/Testing-Cats/"

# Load dataset using custom dataloader
dataset = FramesDataset(path_img)
# Load images from dataset batch-wise using dataloader
dataloader = DataLoader(dataset, batch_size = 16, shuffle = True)

# Show sample batch of images after transformation to tensor objects
images = next(iter(dataloader))
grid_img = torchvision.utils.make_grid(images, nrow=4)
# Convert tensor object to numpy array
npimg = grid_img.permute(1, 2, 0).numpy()
# Plot setup and show image
plt.figure(figsize=(8, 8))
plt.imshow(npimg)
plt.title("Test Images (Pre-Processed)")
plt.axis('off')
# Save sample batch image
plt.savefig(path_out + "test-org.png", dpi=300)



'''VQ-VAE TRAINING LOOP'''



exit()