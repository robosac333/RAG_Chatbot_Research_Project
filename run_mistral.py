from Chatbot.Terminal_Based_Chatbot.Base_Model_with_RAG.Mistral_7b import main

from RAG_Vector_Database.Dataset_to_Vector_Data_Conversion import create_vector_db


# Set pdf folder path
data_path = r"pdfs"

# Create vector database storing path
Db_faiss_path = "Vector_Data_Base_GTR_T5_Large"

# Retrive data from pdfs and store in faiss
create_vector_db(data_path, Db_faiss_path)

# Load the chatbot
main(Db_faiss_path)

