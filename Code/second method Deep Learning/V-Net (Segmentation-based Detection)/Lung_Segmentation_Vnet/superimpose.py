from PIL import Image

# Load the source image and the mask
source_image_path = 'test_src_2.bmp'
mask_image_path = 'test_predict_2.bmp'
output_image_path = 'overlayed_highlight.png'

source_image = Image.open(source_image_path).convert("RGBA")
mask_image = Image.open(mask_image_path).convert("L")  # Load mask as grayscale

# Create an RGBA version of the mask where the white areas will be red and transparent otherwise
mask_rgba = Image.new("RGBA", mask_image.size, (255, 0, 0, 0))  # Initialize with transparent
mask_data = mask_image.getdata()

# Modify the mask to have red color in the regions where it is white
new_data = []
for item in mask_data:
    if item > 0:  # Assuming non-zero mask value means highlighting
        new_data.append((255, 0, 0, 128))  # Red color with 50% transparency
    else:
        new_data.append((255, 0, 0, 0))  # Fully transparent

mask_rgba.putdata(new_data)

# Overlay the mask on the source image
result_image = Image.alpha_composite(source_image, mask_rgba)

# Save or display the result
result_image.save(output_image_path)
result_image.show()
