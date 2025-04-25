# Implementation of Vector-Quantized Variational Auto-Encoder (VAE)
# Inputs:
#   sequence of frames (*.png) located in ./frames/(Game-Name)/

import  matplotlib.pyplot   as plt

import  torchvision

from    torch.utils.data    import DataLoader


# Directory of images
path_img = "./frames/test/"
# Directory for output
path_out = "./output/test/"



'''DATALOADER'''



# Define image transformations to apply to images
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 128)),   # Resize all images to 128x128
    # transforms.Grayscale(),
    torchvision.transforms.ToTensor(),           # Convert PIL Image to tensor [0, 1]
    # transforms.Normalize((0.5,), (0.5,))  # Optional: normalize to [-1, 1]
])

# Load the dataset
dataset = torchvision.datasets.ImageFolder(root = path_img, transform = transform)

# Define the dataloader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)



'''VISUALIZATION SAMPLE INITIAL IMAGES'''



# Select one batch of randomly chosen images
images, labels = next(iter(dataloader))

# Show the images within a grid (as visualization)
grid_img = torchvision.utils.make_grid(images, nrow=4)
npimg = grid_img.permute(1, 2, 0).numpy()  # CHW -> HWC
plt.figure(figsize=(8, 8))
plt.imshow(npimg)
plt.title("Test Images (Pre-Processed)")
plt.axis('off')
plt.savefig(path_out + "test-org.png", dpi=300)



'''VQ-VAE MODEL'''



