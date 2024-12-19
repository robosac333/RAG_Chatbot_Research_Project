from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import os
import fitz  # PyMuPDF
from PIL import Image    
import time

# Define the path to the dataset directory containing PDF and text files
data_path = "/mnt/d/gcodes/RAG_Chatbot_Research_Project/pdfs" 

# Define the path where the FAISS vector database will be saved
Db_faiss_path = "Vector_Data_Base_GTR_T5_Large"

# Custom class to load text files
class TextFileLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    # Method to load text content from the file and wrap it into a Document object
    def load(self):
        print("Reading Text file: ", self.file_path)
        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return [Document(page_content=text, metadata={'source': self.file_path})]

# Function to create a vector database from documents
def create_vector_db(data_path=data_path, Db_faiss_path=Db_faiss_path):
    print("---------------------------------------------------------------")
    
    documents = []  # Initialize a list to hold all documents

    # Load PDF documents from the specified directory using DirectoryLoader
    pdf_loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()  # Load all PDFs into documents
    [print("Retreiving PDF file: ", doc.metadata['source']) for doc in pdf_documents]
    documents.extend(pdf_documents)
    print(len(pdf_documents), "PDF files loaded.")

    # Load all text files in the specified directory
    txt_files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
    for txt_file in txt_files:
        # Use the custom TextFileLoader to load text documents
        txt_loader = TextFileLoader(os.path.join(data_path, txt_file))
        txt_documents = txt_loader.load()
        documents.extend(txt_documents)

    # Split the loaded documents into smaller chunks using a text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(len(texts), "chunks created from the documents.")
    # Create embeddings using a pre-trained HuggingFace model (GTR-T5-Large)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/gtr-t5-large', 
                                        model_kwargs={'device': 'cuda'})
    # Create a FAISS vector store from the document chunks and their embeddings
    db = FAISS.from_documents(texts, embeddings)

    # Save the FAISS database to the specified local path
    db.save_local(Db_faiss_path)

    print("Data Retrived Successfully!")
    print("--------------------------------------")
    print(f"Saved Locally to : {Db_faiss_path}")
    print("--------------------------------------")


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

all_images, file_names = convert_pdfs_to_images(data_path)
# ---------------------------------------------------------------



# ----------------------------------------------------------------
# Seperate the images from the pdf
# ----------------------------------------------------------------
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
# pdf_path = '/fs/nexus-scratch/zahirmd/RAG_Chatbot_Research_Project/Qwen2_ColPali/data/MALM.pdf'
# images_folder = '/fs/nexus-scratch/zahirmd/RAG_Chatbot_Research_Project/pdfs/images'  # Where you want to save the images

# extract_images_from_pdf(data_path, images_folder)

# ---------------------------------------------------------------
# 2. Initialize the ColPali Multimodal Document Retrieval Model
# ---------------------------------------------------------------
from byaldi import RAGMultiModalModel
docs_retrieval_model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
# input the path to the data folder containing the PDFs 
# This creates the indexes of pdfs and assingn a unique name using index_name
docs_retrieval_model.index(input_path=data_path, index_name="image_index", store_collection_with_index=False, overwrite=True)
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


create_vector_db()  # Call the function to create and save the vector database
# Start Mistral 
from Chatbot.Terminal_Based_Chatbot.Base_Model_with_RAG.Mistral_7b import retrieve_faiss, retrieve_context, generate_answer

while True:
    text_query = input("You: ")  # Get user input.
    if text_query.lower() == 'exit':  # Exit if the user types 'exit'.
        print("Goodbye!")
        break
    results = get_relevant_docs(text_query)
    grouped_images = get_grouped_images(results, all_images)


    print(f"No of Relevant Documents Retrieved: {len(results)}")
    print(f"Doc Id: {file_names[results[0]['doc_id']]}, Page Num: {results[0]['page_num']}")
    print(f"Doc Id: {file_names[results[1]['doc_id']]}, Page Num: {results[1]['page_num']}")

    # Read corresponding png image from the folder
    # png_files = [f for f in os.listdir(images_folder) if f.endswith('.png')]
    # image = Image.open(os.path.join(images_folder, png_files[results[0]['doc_id']]))
    # print(f"Extracted Image: {png_files[results[0]['doc_id']]}")


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


    image_context = output_text[0]
    # print("Qwen2 Bot:", output_text[0])  # Display the response.
    # print(f"Time Taken to Respond: {response_time:.2f} seconds")  # Display the response time.

    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # 5. Chat with Chatbot
    # ---------------------------------------------------------------

    db = retrieve_faiss(Db_faiss_path)

    context = retrieve_context(text_query, db)  # Retrieve relevant context documents
    print(f"Image Context: {image_context}")

    answer = generate_answer(text_query, context, image_context=image_context)  # Generate the model's answer

    response_time = time.time() - start_time  # Calculate how long the response took.

    print("Mistral Bot:", answer)  # Display the response.
    print(f"Time Taken to Respond: {response_time:.2f} seconds")  # Display the response time.
    print("----------------------------------------------------------------------")
    # ---------------------------------------------------------------

