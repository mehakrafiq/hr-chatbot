import fitz  # PyMuPDF
from PIL import Image
import os

# Paths
pdf_path = "Pdfs/HRPoliciesAndServiceRules.pdf"  # Input PDF file path
output_folder = "Pdfs"  # Folder to store PNG images
output_pdf_path = "Pdfs/combined_policy_v2.pdf"  # Output combined PDF path

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load PDF
pdf_document = fitz.open(pdf_path)
image_paths = []

# Optional: Set zoom for higher DPI (e.g., 2.0 = 144 DPI, 4.0 = 288 DPI)
apply_zoom = False  # Set to False if zoom is not needed
if apply_zoom:
    zoom_x = 2.5  # Scale factor for x-axis (higher = better quality)
    zoom_y = 2.5  # Scale factor for y-axis (higher = better quality)
    mat = fitz.Matrix(zoom_x, zoom_y)
else:
    mat = fitz.Matrix(1.0, 1.0)  # Default matrix with no zoom

# Loop through each page
for page_num in range(pdf_document.page_count):
    # Load page and convert to image (with optional zoom)
    page = pdf_document.load_page(page_num)
    pix = page.get_pixmap(matrix=mat)  # Apply matrix for zoom or default
    
    # Save each page as a PNG image
    image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
    pix.save(image_path)  # Removed 'format' argument
    image_paths.append(image_path)

# Combine images into a new PDF
image_list = [Image.open(img_path).convert("RGB") for img_path in image_paths]
image_list[0].save(output_pdf_path, save_all=True, append_images=image_list[1:])

# Delete the intermediate PNG files
for img_path in image_paths:
    os.remove(img_path)

print(f"Combined PDF saved as '{output_pdf_path}', intermediate PNG files have been deleted.")
