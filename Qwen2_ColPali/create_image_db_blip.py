# Load the dataset

# ---------------------------------------------------------------
# 1. Convert PDFs to images
# ---------------------------------------------------------------
import os
from pdf2image import convert_from_path
import time


def convert_pdfs_to_images(pdf_folder):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    all_images = {}
    file_names = {}

    for doc_id, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        images = convert_from_path(pdf_path)
        print(f"Converted {pdf_file} to {len(images)} images.")
        all_images[doc_id] = images
        file_names[doc_id] = pdf_file

    return all_images, file_names

all_images, file_names = convert_pdfs_to_images("Qwen2_ColPali/data")
# ---------------------------------------------------------------



# ---------------------------------------------------------------
# 2. Initialize the ColPali Multimodal Document Retrieval Model
# ---------------------------------------------------------------
from byaldi import RAGMultiModalModel
docs_retrieval_model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
# input the path to the data folder containing the PDFs 
# This creates the indexes of pdfs and assingn a unique name using index_name
docs_retrieval_model.index(input_path="Qwen2_ColPali/data", index_name="image_index", store_collection_with_index=False, overwrite=True)
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# 3. Retrieving Documents with the Document Retrieval Model - ColPali
# ---------------------------------------------------------------

def get_relevant_docs(text_query):
    results = docs_retrieval_model.search(text_query, k=2)      # k - Number of relevant document pages to retrieve
    return results


# Get the corresponding images of the retrieved documents
def get_grouped_images(results, all_images):
    grouped_images = []

    for result in results:
        doc_id = result["doc_id"]
        page_num = result["page_num"]
        grouped_images.append(
            all_images[doc_id][page_num - 1]
        )  # page_num are 1-indexed, while doc_ids are 0-indexed. Source https://github.com/AnswerDotAI/byaldi?tab=readme-ov-file#searching

    return grouped_images

# # ---------------------------------------------------------------



# ---------------------------------------------------------------
# 4. Initialize the Visual Language Model 
# ---------------------------------------------------------------
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# 5. Chat with Chatbot
# ---------------------------------------------------------------

while True:
    text_query = input("You: ")
    if text_query.lower() == "exit":
        print("Bye Bye..!")
        break

    results = get_relevant_docs(text_query)
    grouped_images = get_grouped_images(results, all_images)

    print(f"No of Relevant Documents Retrieved: {len(results)}")
    print(f"Doc Id: {file_names[results[0]['doc_id']]}, Page Num: {results[0]['page_num']}")
    print(f"Doc Id: {file_names[results[1]['doc_id']]}, Page Num: {results[1]['page_num']}")

    responses = []
    start_time = time.time()

    for idx, image in enumerate(grouped_images):
        inputs = processor(image, text=text_query, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=100)
        response = processor.decode(outputs[0], skip_special_tokens=True)
        responses.append(response)

    response_time = time.time() - start_time

    # Display the chatbot's response
    print(f"BLIP Bot Response for Image 1: {responses[0]}")
    print(f"BLIP Bot Response for Image 2: {responses[1]}")
    print(f"Time Taken to Respond: {response_time:.2f} seconds")


# ---------------------------------------------------------------

