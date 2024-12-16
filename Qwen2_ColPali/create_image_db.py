# Load the dataset

# ---------------------------------------------------------------
# Convert PDFs to images
# ---------------------------------------------------------------
import os
from pdf2image import convert_from_path


def convert_pdfs_to_images(pdf_folder):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    all_images = {}

    for doc_id, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        images = convert_from_path(pdf_path)
        print(f"Converted {pdf_file} to {len(images)} images.")
        all_images[doc_id] = images

    return all_images

all_images = convert_pdfs_to_images("/mnt/d/gcodes/RAG_Chatbot_Research_Project/Qwen2_ColPali/data")
# ---------------------------------------------------------------



# ---------------------------------------------------------------
# 2. Initialize the ColPali Multimodal Document Retrieval Model
# ---------------------------------------------------------------
from byaldi import RAGMultiModalModel
docs_retrieval_model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
docs_retrieval_model.index(input_path="/mnt/d/gcodes/RAG_Chatbot_Research_Project/Qwen2_ColPali/data", index_name="image_index", store_collection_with_index=False, overwrite=True)
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# 3. Retrieving Documents with the Document Retrieval Model 
# ---------------------------------------------------------------
text_query = "How many people are needed to assemble the Malm?"
results = docs_retrieval_model.search(text_query, k=3)


def get_grouped_images(results, all_images):
    grouped_images = []

    for result in results:
        doc_id = result["doc_id"]
        page_num = result["page_num"]
        grouped_images.append(
            all_images[doc_id][page_num - 1]
        )  # page_num are 1-indexed, while doc_ids are 0-indexed. Source https://github.com/AnswerDotAI/byaldi?tab=readme-ov-file#searching

    return grouped_images


grouped_images = get_grouped_images(results, all_images)
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
    torch_dtype=torch.float16,  # Mixed precision
    quantization_config=bnb_config,  # Apply quantization
)


# print(vl_model.cuda().eval())

min_pixels = 224 * 224
max_pixels = 1024 * 1024
vl_model_processor = Qwen2VLProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
)
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# 5. Assembling
# ---------------------------------------------------------------
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
            {
                "type": "image",
                "image": grouped_images[2],
            },
            {"type": "text", "text": text_query},
        ],
    }
]

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

print(output_text[0])
# ---------------------------------------------------------------

