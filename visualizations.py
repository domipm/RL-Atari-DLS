# Script used for generating different visualization graphs (mainly for testing)

import  os
import  torch
import  torchvision

import  numpy                       as      np
import  matplotlib.pyplot           as      plt

from    PIL                         import  Image
from    torch.utils.data            import  DataLoader
from    torchvision.transforms      import  v2, InterpolationMode

from    vq_vae                      import  VQ_VAE
from    dataloader                  import  FramesDataset, CustomMarginCrop, get_margins



'''PARAMETERS'''



# Game name
fname = "Pong-v5"

# Path to frames
path_frames = f"./frames/{fname}/"
# Path to output files
path_out = f"./output/{fname}/"

# Dimensions of the image
img_dims = ((64, )*2)

# Batch size
batch_size = 16

# Codebook parameters
embedding_num = 512
embedding_dim = 64
beta = 0.25

# Get list of all files
images_fnames = np.sort( os.listdir(path_frames) )

# Open image in grayscale (number 42)
image_org = Image.open(path_frames + images_fnames[42]).convert('L')
# Convert to numpy array
image_org_arr = np.array(image_org)

# Get maximum (latest) model weights
file_weights = []
for file in os.listdir("./output/" + fname + "/"):
    if file.endswith(".pt") and file.startswith("vqvae"):
        file_name = file.split(".")[0].split("_")[2]
        file_weights.append(int(file_name))
file_weights = np.max(file_weights)

# Load model
model = torch.load(f"{path_out}vqvae_weights_{file_weights}.pt", weights_only = False)
model.eval()  # Set model to evaluation mode

# Compute margins based on dynamic map of frames
motion, l, r, t, b = get_margins(path_frames = path_frames)

# Transform object to apply to frames
transform_org = v2.Compose([
    # Convert to grayscale
    v2.Grayscale(),
    # Convert to tensor object
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
])

# Transform object to apply to frames
transform_pre = v2.Compose([
    # Convert to grayscale
    v2.Grayscale(),
    # Custom crop
    CustomMarginCrop(l, r, t, b),
    # Convert to tensor object
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    # Resize all images (square shape or divisible by 2!)
    v2.Resize( img_dims , interpolation = InterpolationMode.NEAREST_EXACT, antialias = False),
])

# Load dataset (no transform)
dataset_org = FramesDataset("./frames/" + fname, transform_org)
# Load dataset (pre-processed)
dataset_pre = FramesDataset("./frames/" + fname, transform_pre)

# Create batch dataloader (pre-processed)
dataloader_pre = DataLoader(dataset_pre, batch_size = 16, shuffle = True)



'''CODEBOOK'''



# Load batch of images
image = next(iter(dataloader_pre))

# Initialize autoencoder
autoenc = VQ_VAE(embedding_num, embedding_dim, beta, image.shape)
# Set evaluation mode
autoenc.eval()
# Ensure no gradients computed
with torch.no_grad():
    # Preprocess, encode, and quantize image
    z_e, indices, _ = autoenc.encode_and_quantize(x = image)

# Reshape back to 2D
indices_re = indices.view(batch_size, z_e.shape[2], z_e.shape[2]).detach().numpy()

n_img = 4
n_col = 2

fig, ax = plt.subplots(nrows = n_col, ncols = n_img)

# Plot originals
for i in range(n_img):
        ax[0,i].imshow(image[i][0], cmap = 'viridis')
        ax[0,i].axis('off')
# Plot codebook 
for i in range(n_img):
        ax[1,i].imshow(indices_re[i], cmap = 'viridis')
        ax[1,i].axis('off')

plt.tight_layout()
plt.savefig(f"./output/{fname}/vqvae_codebook.pdf")

plt.close()



'''PREPROCESSING'''



# Setup plots
fig, ax = plt.subplots(nrows = 1, ncols = 3)

# Remove axis lines
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')

# Titles
ax[0].set_title("Original Img.")
ax[1].set_title("Motion Map & Margins")
ax[2].set_title("Pre-Processed Img.")

# Plot
ax[0].imshow(dataset_org[42][0], cmap = "gray")
ax[1].imshow(motion, cmap = "gray")
ax[2].imshow(dataset_pre[42][0], cmap = "gray")

# Boundary lines
ax[1].vlines(x = l, ymin = 0, ymax = 210, color="green", linewidth=1)
ax[1].vlines(x = r, ymin = 0, ymax = 210, color="green", linewidth=1)

ax[1].vlines(x = l - 5, ymin = 0, ymax = 210, color="red", linewidth=1)
ax[1].vlines(x = r + 5, ymin = 0, ymax = 210, color="red", linewidth=1)

ax[1].hlines(y = b, xmin = 0, xmax = 160, color="green")
ax[1].hlines(y = b + 5, xmin = 0, xmax = 160, color="red")

ax[1].hlines(y = t, xmin = 0, xmax = 160, color="green")
ax[1].hlines(y = t - 5, xmin = 0, xmax = 160, color="red")

# Save figure
plt.tight_layout()
plt.savefig(f"{path_out}vqvae_preprocess.pdf")
plt.show()



'''EVALUATION / RECONSTRUCTION'''



# Setup plots
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 8))

# Remove axis lines
ax[0].axis('off')
ax[1].axis('off')
# Titles
ax[0].set_title("Original Images")
ax[1].set_title("Reconstructed Images")

# Show sample batch of images after transformation to tensor objects
images = next(iter(dataloader_pre))

# Convert batch of images into grid
grid_img = torchvision.utils.make_grid(images.detach(), nrow=4)
# Convert tensor object to numpy array
npimg    = grid_img.permute(1, 2, 0).numpy()

# Plot image
ax[0].imshow(npimg)

# Set model to evaluation mode
model.eval()

# Obtain reconstructed images from the model
reconstructed, _, _ = model(images)

# Convert batch of images into grid
grid_img = torchvision.utils.make_grid(reconstructed.detach(), nrow=4)
# Convert tensor object to numpy array
npimg    = grid_img.permute(1, 2, 0).numpy()

# Plot image
ax[1].imshow(npimg)

# Setup narrow margins and save image
plt.tight_layout()
plt.savefig(path_out + "vqvae_sample.pdf")
# Close plot
plt.close()