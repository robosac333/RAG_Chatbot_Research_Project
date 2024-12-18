# Load the dataset

# ---------------------------------------------------------------
# 1. Convert PDFs to images
# ---------------------------------------------------------------
import os
from pdf2image import convert_from_path
import time
import numpy as np
import cv2

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


all_images, file_names = convert_pdfs_to_images("/fs/nexus-scratch/zahirmd/RAG_Chatbot_Research_Project/pdfs")
# ---------------------------------------------------------------



# ---------------------------------------------------------------
# 2. Initialize the ColPali Multimodal Document Retrieval Model
# ---------------------------------------------------------------
from byaldi import RAGMultiModalModel
docs_retrieval_model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
# input the path to the data folder containing the PDFs 
# This creates the indexes of pdfs and assingn a unique name using index_name
docs_retrieval_model.index(input_path="/fs/nexus-scratch/zahirmd/RAG_Chatbot_Research_Project/pdfs", index_name="image_index", store_collection_with_index=False, overwrite=True)
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# 3. Retrieving Documents with the Document Retrieval Model - ColPali
# ---------------------------------------------------------------

def get_relevant_docs(text_query):
    results = docs_retrieval_model.search(text_query, k=3)      # k - Number of relevant document pages to retrieve
    return results


# Get the corresponding images of the retrieved documents
def get_grouped_images(results, all_images):
    grouped_images = []

    for result in results:
        doc_id = result["doc_id"]
        page_num = result["page_num"]
        # print(result.keys())
        grouped_images.append(
            all_images[doc_id][page_num - 1]
        )  # page_num are 1-indexed, while doc_ids are 0-indexed. Source https://github.com/AnswerDotAI/byaldi?tab=readme-ov-file#searching

    return grouped_images

# # ---------------------------------------------------------------



# ---------------------------------------------------------------
# 4. Initialize the Visual Language Model 
# ---------------------------------------------------------------
from transformers import pipeline, AutoProcessor
from PIL import Image    
import requests

model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline("image-to-text", model=model_id, device=0)
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
    # grouped_images = get_grouped_images(results, all_images)
    print(f"No of Relevant Documents Retrieved: {len(results)}")

    # Get the corresponding image from the folder
    images_path = "/fs/nexus-scratch/zahirmd/RAG_Chatbot_Research_Project/pdfs"
    # Read 4th png image from the folder
    png_files = [f for f in os.listdir(images_path) if f.endswith('.png')]
    image = Image.open(os.path.join(images_path, png_files[results[0]['doc_id']]))

    print(f"Extracted Image: {png_files[results[0]['doc_id']]}")


    # grouped_images = [cv2.resize(np.array(image), (640, 640)) for image in grouped_images]
    # grouped_images_np = [np.array(image) for image in grouped_images]
    # print(grouped_images_np[0].shape, grouped_images_np[1].shape, grouped_images_np[2].shape)
    # concatenated_image = np.concatenate((grouped_images_np[0], grouped_images_np[1], grouped_images_np[2]), axis=1)


    # print(f"No of Relevant Documents Retrieved: {len(results)}")
    # print(f"Doc Id: {file_names[results[0]['doc_id']]}, Page Num: {results[0]['page_num']}")

    # If there are less than 3 images, consider the full document
    # if len(all_images[results[0]['doc_id']]) < 3:
    #     grouped_images = [cv2.resize(np.array(image), (1080, 1080)) for image in all_images[results[0]['doc_id']]]
    #     concatenated_image = np.concatenate(grouped_images, axis=1)


    # Concatenate the images horizontally (you can also concatenate vertically if needed)

    # Convert the concatenated image back to a PIL Image
    # concatenated_image = Image.fromarray(concatenated_image)
    # Chat template for Qwen2 Model
    chat_template = [
        {
            "role": "user",
            "content": [
                # {
                    # "type": "image",
                    # "image": grouped_images[0],
                # },
                # {
                    # "type": "image",
                    # "image": grouped_images[1],
                # },
                {
                    "type": "image",
                    # "image": grouped_images[2],
                },
                {"type": "text", "text": text_query},
            ],
        }
    ]

    # Query the model
    start_time = time.time()  # Track the start time for response timing.

    processor = AutoProcessor.from_pretrained(model_id)

    prompt = processor.apply_chat_template(chat_template, add_generation_prompt=True)

    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

    response_time = time.time() - start_time  # Calculate how long the response took.

    print("----------------------------------------------------------------------")
    print("Llava Bot:", outputs[0]["generated_text"].split("ASSISTANT:")[1])  # Display the response.
    print("----------------------------------------------------------------------")
    
    print(f"Time Taken to Respond: {response_time:.2f} seconds")  # Display the response time.


# ---------------------------------------------------------------

