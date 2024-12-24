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

all_images, file_names = convert_pdfs_to_images("/nfshomes/sjd3333/RAG_Chatbot_Research_Project/img_pdfs")
# ---------------------------------------------------------------



# ---------------------------------------------------------------
# 2. Initialize the ColPali Multimodal Document Retrieval Model
# ---------------------------------------------------------------
from byaldi import RAGMultiModalModel
docs_retrieval_model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
# input the path to the data folder containing the PDFs 
# This creates the indexes of pdfs and assingn a unique name using index_name
docs_retrieval_model.index(input_path="RAG_Chatbot_Research_Project/img_pdfs", index_name="image_index", store_collection_with_index=False, overwrite=True)
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
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
import torch
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Use 4-bit quantization
)

vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    # torch_dtype=torch.float16,  # Mixed precision
    torch_dtype=torch.bfloat16,  # Mixed precision
    quantization_config=bnb_config,  # Apply quantization
)

min_pixels = 224 * 224
max_pixels = 1024 * 1024
vl_model_processor = Qwen2VLProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
)
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

    # Chat template for Qwen2 Model
    chat_template = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": grouped_images[0],
                },
                {
                    "type": "image",
                    "image": grouped_images[1],
                },
                # {
                #     "type": "image",
                #     "image": grouped_images[2],
                # },
                {"type": "text", "text": text_query},
            ],
        }
    ]

    # Query the model
    start_time = time.time()  # Track the start time for response timing.

    text = vl_model_processor.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)

    image_inputs, _ = process_vision_info(chat_template)
    inputs = vl_model_processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = vl_model.generate(**inputs, max_new_tokens=500)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = vl_model_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    response_time = time.time() - start_time  # Calculate how long the response took.

    print("Qwen2 Bot:", output_text[0])  # Display the response.
    
    print(f"Time Taken to Respond: {response_time:.2f} seconds")  # Display the response time.


# ---------------------------------------------------------------

