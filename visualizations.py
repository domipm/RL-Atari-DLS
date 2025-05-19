# Script used for generating different visualization graphs (mainly for testing)

import  os
import  torch

import  numpy                       as      np
import  matplotlib.pyplot           as      plt

from    PIL                         import  Image
from    torch.utils.data            import  DataLoader, random_split
from    torchvision.transforms      import  v2, InterpolationMode

from    vq_vae                      import  VQ_VAE
from    dataloader                  import  FramesDataset, CustomMarginCrop, get_margins



# Game name
fname = "Boxing-v5"

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

# Load model
model = torch.load(f"{path_out}vqvae_weights_50.pt", weights_only = False)
model.eval()  # Set model to evaluation mode

# Compute margins based on dynamic map of frames
motion, l, r, t, b = get_margins(path_frames = path_frames)

# Transform object to apply to frames
transform_frames = v2.Compose([
    # Convert to grayscale
    v2.Grayscale(),
    # Custom crop
    CustomMarginCrop(l, r, t, b),
    # Convert to tensor object
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    # Resize all images (square shape or divisible by 2!)
    v2.Resize( img_dims , interpolation = InterpolationMode.NEAREST, antialias = False),
])

# Load dataset using custom dataloader
dataset = FramesDataset("./frames/" + fname, transform_frames)

# Load train images from dataset batch-wise using dataloader
dataloader_train = DataLoader(dataset, batch_size = 16, shuffle = False)

# Take batch
image = next(iter(dataloader_train)) # .unsqueeze(0).repeat(16, 1, 1, 1)



'''CODEBOOK'''



# Initialize autoencoder
autoenc = VQ_VAE(embedding_num, embedding_dim, beta, image.shape)

autoenc.eval()

with torch.no_grad():
    z_e, indices, _ = autoenc.encode_and_quantize(x = image)

# Reshape back to 2D
indices_re = indices.view(batch_size, z_e.shape[0], z_e.shape[0]).detach().numpy()

nrow = 4

B, H, W = indices_re.shape
ncol = (B + nrow - 1) // nrow  # Compute number of rows based on total batch size

fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 3, ncol * 3))
axes = axes.flatten()

for i in range(B):
    ax = axes[i]
    ax.imshow(indices_re[i], cmap='viridis')
    ax.set_title(f"Sample {i}")
    ax.axis('off')

# Hide unused subplots
for j in range(B, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig(f"./output/{fname}/vqvae_codebook.pdf")
# plt.show()
plt.close()



'''PREPROCESSING'''



# Setup plots
fig, ax = plt.subplots(nrows = 1, ncols = 3)

# Compute pre-processed image
image_post = dataset[42].detach().numpy()[0]

# Remove axis lines
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')

# Titles
ax[0].set_title("Original Img.")
ax[1].set_title("Motion Map & Margins")
ax[2].set_title("Pre-Processed Img.")

# Plot
ax[0].imshow(image_org_arr, cmap = "gray")
ax[1].imshow(motion, cmap = "gray")
ax[2].imshow(image_post, cmap = "gray")

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