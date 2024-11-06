import torch

# Importing necessary libraries and modules from Hugging Face's `peft` and `transformers` for model handling, tokenization, and fine-tuning.
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, TrainingArguments
import os
import time
from langchain.vectorstores import FAISS  # For working with vector stores
from langchain.embeddings.huggingface import HuggingFaceEmbeddings  # For generating embeddings using Hugging Face models
from langchain.schema import Document  # For handling document schema

# Load the tokenizer from a pre-trained GPTQ model and set the padding token to be the end-of-sequence token.
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GPTQ")
tokenizer.pad_token = tokenizer.eos_token

# Load a quantization configuration for GPTQ, which reduces model precision to 4-bits for efficient inference.
quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True, tokenizer=tokenizer)

# Load the pre-trained model with the quantization configuration, automatically assigning the model to the available devices.
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",
    quantization_config=quantization_config_loading,
    device_map="auto"
)

# Disable caching of attention layers to save memory and enable gradient checkpointing to reduce memory usage during training.
model.config.use_cache = False
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# Prepare the model for 8-bit (k-bit) training using LoRA (Low-Rank Adaptation) for efficient fine-tuning.
model = prepare_model_for_kbit_training(model)

# Set up a LoRA configuration, targeting specific layers of the model (`q_proj`, `v_proj`) for fine-tuning.
peft_config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)

# Configure generation settings for the model, including sampling strategies (e.g., `do_sample`, `top_k`, `temperature`) and maximum token length.
generation_config = GenerationConfig(
    do_sample=True,
    top_k=1,
    temperature=0.1,
    max_new_tokens=100,
    pad_token_id=tokenizer.eos_token_id
)

# Define a system prompt to instruct the model's behavior during interaction, focusing on providing helpful and accurate responses.
system_prompt = """
You are a helpful and informative assistant. Your goal is to answer questions accurately, thoroughly, and naturally. Provide detailed explanations and context when possible. If you do not understand or do not have enough information to answer, simply say - "Sorry, I don't know." Avoid formatting your response as a multiple-choice answer.
"""

# Initialize embeddings using a pre-trained sentence transformer model (`gtr-t5-large`) for query processing and document retrieval.
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/gtr-t5-large', 
                                   model_kwargs={'device': 'cuda'})

# Load a FAISS vector database for efficient similarity search, using embeddings generated by the sentence transformer model.
Db_faiss_path = r"C:\Users\Computing\Downloads\Two Models\Flan-T5-Model\Flan-T5-Model\Vector_Data_Base_GTR_T5_Large"
db = FAISS.load_local(Db_faiss_path, embeddings, allow_dangerous_deserialization=True)

# Function to retrieve the most relevant documents from the FAISS database given a query, returning the top `k` results.
def retrieve_context(query, k=10):
    query_embedding = embeddings.embed_query(query)
    docs = db.similarity_search_by_vector(query_embedding, k)
    ranked_docs = rank_documents(query, docs)
    return ranked_docs[:5]

# Function to rank documents based on their relevance to the query; currently a placeholder that returns the docs unmodified.
def rank_documents(query, docs):
    return docs 

# Function to generate a response from the model based on the query and retrieved context.
def generate_answer(query, context):
    # Combine the context documents into a single text block to include in the prompt.
    context_text = "\n\n".join([doc.page_content for doc in context])
    
    # Create a structured input for the model with a clear prompt, context, and user query.
    input_text = (
        system_prompt +
        "\nContext: " + context_text + 
        "\n\nUser: " + query + 
        "\nBot:"
    )
    
    # Tokenize the input text and move it to the GPU for processing.
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    # Generate a response using the configured generation settings.
    outputs = model.generate(**inputs, generation_config=generation_config)
    
    # Decode the generated tokens into a text response, removing special tokens and extraneous information.
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Return the clean answer, stripping out unnecessary prefixes or suffixes.
    return answer.strip().split("\nBot:")[-1].strip()

# Function to handle the entire response generation process, including context retrieval, model inference, and response formatting.
def get_response(question):
    context = retrieve_context(question)  # Retrieve relevant context documents.
    start_time = time.time()  # Track the start time for response timing.
    answer = generate_answer(question, context)  # Generate the model's answer.
    response_time = time.time() - start_time  # Calculate how long the response took.
    sources = [doc.metadata['source'] for doc in context]  # Extract sources for transparency.
    return answer, sources, response_time  # Return the answer, sources, and response time.

# Main function to run the chatbot interface in a loop until the user decides to exit.
def main():
    print("Welcome to the chatbot. Type your question and press enter.")
    print("Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")  # Get user input.
        if user_input.lower() == 'exit':  # Exit if the user types 'exit'.
            print("Goodbye!")
            break
        response, sources, response_time = get_response(user_input)  # Generate the response.
        print("Bot:", response)  # Display the response.
        # print("Sources:", sources)  # Optionally display the sources (currently commented out).
        print(f"Time Taken to Respond: {response_time:.2f} seconds")  # Display the response time.

# Entry point for the script, starting the chatbot.
if __name__ == "__main__":
    main()
