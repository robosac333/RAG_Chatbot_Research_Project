import fitz  # PyMuPDF
import os

def extract_images_from_pdf(pdf_path, output_folder):
    # Open the PDF
    doc = fitz.open(pdf_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img_count = 0  # To count the number of images extracted

    # Iterate through the pages of the PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # Get the page
        
        # Extract images from the page
        image_list = page.get_images(full=True)
        
        # Loop through each image on the page
        for img_index, img in enumerate(image_list):
            xref = img[0]  # xref of the image
            base_image = doc.extract_image(xref)  # Extract the image as a dictionary
            image_bytes = base_image["image"]  # Get the image bytes
            
            # Create an output path for saving the image
            img_filename = f"image_{img_count + 1}.png"
            img_path = os.path.join(output_folder, img_filename)
            
            # Save the image to the output folder
            with open(img_path, "wb") as img_file:
                img_file.write(image_bytes)
            
            img_count += 1

    print(f"Extracted {img_count} images and saved them to {output_folder}")

# Example usage
pdf_path = '/fs/nexus-scratch/zahirmd/RAG_Chatbot_Research_Project/Qwen2_ColPali/data/MALM.pdf'
output_folder = '/fs/nexus-scratch/zahirmd/RAG_Chatbot_Research_Project/pdfs'  # Where you want to save the images

extract_images_from_pdf(pdf_path, output_folder)
