from Chatbot.Terminal_Based_Chatbot.Base_Model_with_RAG.Mistral_7b import main

from RAG_Vector_Database.Dataset_to_Vector_Data_Conversion import create_vector_db

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Template

app = FastAPI()

# Set pdf folder path
data_path = r"/mnt/d/gcodes/RAG_Chatbot_Research_Project/pdfs"

# Create vector database storing path
Db_faiss_path = "Vector_Data_Base_GTR_T5_Large_new"

# Retrive data from pdfs and store in faiss
create_vector_db(data_path, Db_faiss_path)

# Load the chatbot
main(Db_faiss_path)

@app.get("/")
def read_root():
    return {"Hello": "World"}

