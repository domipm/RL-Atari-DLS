import  os
import  torch

import  numpy   as  np

from    PIL                     import  Image

from    torch.utils.data        import  Dataset

from    torchvision.transforms  import  v2
from    torchvision.transforms  import  functional  as  F



'''CUSTOM FRAME DATASET LOADER'''



# Custom class for loading frames
class FramesDataset(Dataset):


    # Initialization function for dataset
    def __init__(self, directory, transform = None):
        # Initialize parent class
        super().__init__()

        # Initialize directory
        self.directory = directory
        # List all images within directory (ignoring ".*" files)
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



'''COMPUTE MARGINS TO CROP FRAMES TO'''



# Function to compute size of dynamic image
# (size of unused margins, scoreboards, etc.)
def get_margins(path_frames):
    '''
    Function used to compute binary motion map for 
    given frame dataset and extract the margins to which
    images can be cropped.

    Input:
        path_frames: Path to frames
    Returns:
        cdiff_bin: Binary cumulative difference or motion map
        lmargin, rmargin, tmargin, bmargin: left, right, top, and bottom margins
    '''

    # Threshold for ignoring binary pixels
    pnum = 40

    # Get list of all files in directory
    images_fnames = np.sort( os.listdir(path_frames) )

    # Open one image as sample
    image_sample_pil = Image.open(path_frames + images_fnames[np.random.choice(range(len(images_fnames)))]).convert('L')
    # Convert to numpy array
    image_sample_arr = np.array(image_sample_pil)
    # Close pil image
    image_sample_pil.close()

    # Get dimensions of image (Height, Width)
    H, W = image_sample_arr.shape

    # Zeros array same dimensions as images
    cdiff = np.zeros_like(image_sample_arr, dtype = np.float64)

    # Compute consecutive distance between frames
    for k in range(len(images_fnames) - 1):

        # Open image in grayscale
        image_pil = Image.open(path_frames + images_fnames[k]).convert('L')
        # Open next image in grayscale
        image_next_pil = Image.open(path_frames + images_fnames[k + 1]).convert('L')

        # Convert to numpy arrays
        image_arr = np.array(image_pil)
        image_next_arr = np.array(image_next_pil)

        # Compute cumulative difference
        cdiff += np.abs( image_arr - image_next_arr )

        # Close pil images
        image_pil.close()
        image_next_pil.close()

    # Convert to binary motion map
    cdiff_bin = np.zeros_like(cdiff, dtype = int)
    cdiff_bin[np.where(cdiff != 0)] = 1

    # Compute margins

    lmargin = 0
    # Find left vertical margin
    for x in range( 0, cdiff.shape[1] // 2, 1 ):
        if np.any(cdiff_bin[:, x]) == 0:
            lmargin = x
        else:
            break
    rmargin = W
    # Find right vertical margin
    for x in range( cdiff.shape[1] - 1, cdiff.shape[1] // 2, -1 ):
        if np.any(cdiff_bin[:, x]) == 0:
            rmargin = x
        else:
            break
    # Find top horizontal margin
    tmargin = 0
    for y in range( 0, cdiff.shape[0] // 2, 1 ):
        if np.any(cdiff_bin[y, :]) == 0:
            tmargin = y
        else:
            if np.sum(cdiff_bin[y, :]) < pnum:
                tmargin = y
            else:
                break
    bmargin = H - 1
    # Find bottom horizontal margin
    for y in range( cdiff.shape[0] - 1, cdiff.shape[0] // 2, -1 ):
        if np.any(cdiff_bin[y, :]) == 0:
            bmargin = y
        else:
            break
    
    # Return margins
    return cdiff_bin, lmargin, rmargin, tmargin, bmargin



'''Custom Cropping Class'''



# Crop to use with v2.Compose transforms
class CustomMarginCrop:
    
    def __init__(self, left=0, right=0, top=0, bottom=0, pad = 2, in_shape = [0, 0]):
        # Define shape of margins
        # self.left = left
        # self.right = right
        # self.top = top
        # self.bottom = bottom

        # Compute shape of final image (check if padding can be applied safely)

        # Top margin
        if top - pad > 0:
            self.top = top - pad
        else:
            self.top = top
        # Left margin
        if left - pad > 0:
            self.left = left - pad
        else:
            self.left = left
        # "Width" new dimension
        if right + pad < in_shape[1]:
            self.width = right - self.left + pad
        else:
            self.width = right - self.left
        # "Height" new dimension
        if bottom + pad < in_shape[0]:
            self.height = bottom - self.top + pad
        else:
            self.height = bottom - self.top
        

    def __call__(self, img):
        # Perform the cropping (try with padding)

        return F.crop(img,
                      top    = self.top,
                      left   = self.left,
                      width  = self.width,
                      height = self.height, )