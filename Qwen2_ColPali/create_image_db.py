# Load the dataset

# ---------------------------------------------------------------
# Convert PDFs to images
# ---------------------------------------------------------------
# import os
# from pdf2image import convert_from_path


# def convert_pdfs_to_images(pdf_folder):
#     pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
#     all_images = {}

#     for doc_id, pdf_file in enumerate(pdf_files):
#         pdf_path = os.path.join(pdf_folder, pdf_file)
#         images = convert_from_path(pdf_path)
#         print(f"Converted {pdf_file} to {len(images)} images.")
#         all_images[doc_id] = images

#     return all_images

# all_images = convert_pdfs_to_images("/mnt/d/gcodes/RAG_Chatbot_Research_Project/Qwen2_ColPali/data")
# ---------------------------------------------------------------



# # ---------------------------------------------------------------
# # 2. Initialize the ColPali Multimodal Document Retrieval Model
# # ---------------------------------------------------------------
# from byaldi import RAGMultiModalModel
# docs_retrieval_model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
# docs_retrieval_model.index(input_path="/mnt/d/gcodes/RAG_Chatbot_Research_Project/Qwen2_ColPali/data", index_name="image_index", store_collection_with_index=False, overwrite=True)
# # ---------------------------------------------------------------


# # ---------------------------------------------------------------
# # 3. Retrieving Documents with the Document Retrieval Model 
# # ---------------------------------------------------------------
# text_query = "How many people are needed to assemble the Malm?"
# results = docs_retrieval_model.search(text_query, k=3)
# # ---------------------------------------------------------------



# ---------------------------------------------------------------
# 4. Initialize the Visual Language Model 
# ---------------------------------------------------------------
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
import torch

vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
)
vl_model.cuda().eval()