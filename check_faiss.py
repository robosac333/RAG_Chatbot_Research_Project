import faiss
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Path to your FAISS index
faiss_index_path = "/mnt/d/gcodes/RAG_Chatbot_Research_Project/Vector_Data_Base_GTR_T5_Large"

# Initialize the embedding model (use the same model you used to create the index)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/gtr-t5-large')

# Load the FAISS index
vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

# Get the number of vectors in the index
num_vectors = vector_store.index.ntotal
print(f"Number of vectors in the index: {num_vectors}")

# Retrieve and print some sample documents
sample_size = min(5, num_vectors)  # Adjust the sample size as needed
sample_docs = vector_store.similarity_search("", k=sample_size)

print("\nSample documents:")
for i, doc in enumerate(sample_docs, 1):
    print(f"\nDocument {i}:")
    print(f"Content: {doc.page_content[:200]}...")  # Print first 200 characters
    print(f"Metadata: {doc.metadata}")

# Optional: Print index statistics
if isinstance(vector_store.index, faiss.IndexFlatL2):
    print(f"\nIndex type: Flat L2")
elif isinstance(vector_store.index, faiss.IndexIVFFlat):
    print(f"\nIndex type: IVF Flat")
    print(f"Number of centroids: {vector_store.index.nlist}")
    print(f"Number of probes: {vector_store.index.nprobe}")
