import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from PIL import Image
import tifffile as tiff

# Memory-map the large TIFF file (replace with your actual file path)
tif_file = 'path_to_tif_file'
with tiff.TiffFile(tif_file) as tif:
    img = tif.asarray(out='memmap')  # Memory-map the image stack

# Get the total number of frames (slices) in the stack
frame_count = img.shape[0]

# Function to display a specific frame lazily (only load the required frame)
def show_frame(frame_idx):
    # Access only the specific slice from the memory-mapped array
    part_img = img[frame_idx]
    
    # Convert the numpy array slice to a PIL image
    pil_img = Image.fromarray(part_img)
    
    # Display the image
    plt.imshow(pil_img, cmap='gray')
    plt.axis('off')
    plt.show()

# Interactive slider to scroll through frames without loading them all
interact(show_frame, frame_idx=IntSlider(min=0, max=frame_count-1, step=1, value=0))