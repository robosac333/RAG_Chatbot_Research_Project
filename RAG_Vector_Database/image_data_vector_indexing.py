from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import os
import fitz  # PyMuPDF
import PIL.Image
import io
import json
from transformers import AutoProcessor, AutoModel
import torch
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class DocumentSegment:
    text: str
    page_num: int
    images: List[Dict[str, Any]]
    bbox: tuple  # Bounding box coordinates

class EnhancedPDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        # Initialize CLIP-like model for image understanding
        self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        if torch.cuda.is_available():
            self.image_model = self.image_model.to('cuda')

    def extract_images_and_text(self, pdf_document):
        segments = []
        for page_num, page in enumerate(pdf_document):
            # Extract text with positions
            text_blocks = page.get_text("blocks")
            
            # Extract images
            images = self._extract_page_images(page)
            
            # Process each text block and associate with nearby images
            for block in text_blocks:
                bbox = block[:4]  # Bounding box coordinates
                text = block[4]
                
                # Find images that are spatially close to this text block
                related_images = self._find_related_images(images, bbox)
                
                segment = DocumentSegment(
                    text=text,
                    page_num=page_num,
                    images=related_images,
                    bbox=bbox
                )
                segments.append(segment)
                
        return segments

    def _extract_page_images(self, page):
        images = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            
            if base_image:
                # Convert image data to PIL Image
                image_data = base_image["image"]
                image = PIL.Image.open(io.BytesIO(image_data))
                
                # Get image position on page
                rect = page.get_image_bbox(xref)
                
                # Generate image embedding using CLIP
                image_embedding = self._generate_image_embedding(image)
                
                images.append({
                    "image": image,
                    "bbox": rect,
                    "embedding": image_embedding,
                    "index": img_index
                })
                
        return images

    def _generate_image_embedding(self, image):
        with torch.no_grad():
            inputs = self.image_processor(images=image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            image_features = self.image_model.get_image_features(**inputs)
            return image_features.cpu().numpy()

    def _find_related_images(self, images, text_bbox, proximity_threshold=100):
        related_images = []
        
        for img in images:
            # Calculate spatial distance between text and image
            distance = self._calculate_bbox_distance(text_bbox, img["bbox"])
            
            if distance < proximity_threshold:
                related_images.append({
                    "index": img["index"],
                    "bbox": img["bbox"],
                    "embedding": img["embedding"].tolist()
                })
                
        return related_images

    def _calculate_bbox_distance(self, bbox1, bbox2):
        # Calculate center points
        center1 = ((bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2)
        center2 = ((bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2)
        
        # Calculate Euclidean distance
        return ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5

    def load(self):
        doc = fitz.open(self.file_path)
        segments = self.extract_images_and_text(doc)
        
        # Convert segments to Documents for Langchain
        documents = []
        for segment in segments:
            # Create metadata including image information
            metadata = {
                'source': self.file_path,
                'page': segment.page_num,
                'bbox': segment.bbox,
                'images': segment.images
            }
            
            documents.append(Document(
                page_content=segment.text,
                metadata=metadata
            ))
            
        return documents

def create_vector_db(data_path, Db_faiss_path):
    documents = []

    # Load PDF documents with enhanced image processing
    for pdf_file in [f for f in os.listdir(data_path) if f.endswith('.pdf')]:
        pdf_loader = EnhancedPDFLoader(os.path.join(data_path, pdf_file))
        pdf_documents = pdf_loader.load()
        documents.extend(pdf_documents)

    # Load text files (unchanged from original)
    txt_files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
    for txt_file in txt_files:
        txt_loader = TextFileLoader(os.path.join(data_path, txt_file))
        txt_documents = txt_loader.load()
        documents.extend(txt_documents)

    # Split documents while preserving image metadata
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
    )
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/gtr-t5-large',
        model_kwargs={'device': 'cuda'}
    )

    # Create and save vector database
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(Db_faiss_path)

if __name__ == '__main__':
    create_vector_db()