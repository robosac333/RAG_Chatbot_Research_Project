from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import os

# Define the path to the dataset directory containing PDF and text files
data_path = "/nfshomes/sjd3333/RAG_Chatbot_Research_Project/Files_to_process" 
Db_faiss_path = "/nfshomes/sjd3333/RAG_Chatbot_Research_Project/Vector_Data_Base_GTR_T5_Large"

class TextFileLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        documents = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.txt') or file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    print(f"Processing text file: {file_path}")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            text = file.read()
                        documents.append(Document(page_content=text, metadata={'source': self.file_path}))
                    except Exception as e:
                        print(f"Error processing file {file_path}: {str(e)}")
        return documents

def create_vector_db():
    documents = []

    # Load PDF documents
    try:
        print("Loading PDF documents...")
        pdf_loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
        pdf_documents = pdf_loader.load()
        print(f"Loaded {len(pdf_documents)} PDF documents")
        documents.extend(pdf_documents)
    except Exception as e:
        print(f"Error loading PDFs: {str(e)}")

    # Load text files
    try:
        print("Loading text files...")
        txt_files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
        print(f"Found {len(txt_files)} text files")
        
        for txt_file in txt_files:
            txt_loader = TextFileLoader(os.path.join(data_path, txt_file))
            txt_documents = txt_loader.load()
            if txt_documents:
                documents.extend(txt_documents)
    except Exception as e:
        print(f"Error loading text files: {str(e)}")

    print(f"Total documents loaded: {len(documents)}")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} text chunks")

    # Create embeddings
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/gtr-t5-large',
        model_kwargs={'device': 'cuda'}
    )

    # Create and save FAISS database
    print("Creating FAISS database...")
    if not texts:
        raise ValueError("No texts to process! Check your input files.")
    
    db = FAISS.from_documents(texts, embeddings)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(Db_faiss_path), exist_ok=True)
    
    # Save the database
    print(f"Saving database to {Db_faiss_path}")
    db.save_local(Db_faiss_path)
    
    # Verify the save
    if os.path.exists(Db_faiss_path):
        print(f"Vector database saved successfully at {Db_faiss_path}")
        print(f"Database size: {sum(os.path.getsize(os.path.join(Db_faiss_path, f)) for f in os.listdir(Db_faiss_path))} bytes")
    else:
        print("Warning: Database directory not found after save!")

if __name__ == '__main__':
    create_vector_db()