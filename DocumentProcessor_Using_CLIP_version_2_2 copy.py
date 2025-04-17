# new package to add to requirement are scikit-learn, requests, pdfplumber
import os
import math
import chromadb
import chromadb.api
import re
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch
from langchain_community.vectorstores.utils import filter_complex_metadata
import cv2
import pdfplumber
import fitz  # PyMuPDF
from io import BytesIO
from PIL import Image
from docx import Document as DocxDocument
from docx.shared import Inches
# import pytesseract
# import openai
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
from openai import OpenAI
# import json
from dotenv import load_dotenv
# import requests

#setting up the OpenAI API key 
client = OpenAI()
chroma_client = chromadb.PersistentClient(path=r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\VectorDB")# can also use server local
path= r"C:\Users\DELL\Desktop\Chatbot\My_chat_bot\Transformers"
# Initialize CLIP model and processor from Hugging Face
clip_model = CLIPModel.from_pretrained(path)
clip_processor = CLIPProcessor.from_pretrained(path)
device = "cpu"
clip_model.to(device)

class CSVProcessor:
    def __init__(self, file_path, persist_directory='persist_chroma_csv'):
        self.file_path = file_path
        self.persist_directory = persist_directory
        self.loader = CSVLoader(file_path=self.file_path, encoding='utf-8', csv_args={'delimiter': '|'})
        self.embedding = OpenAIEmbeddings()

    def process(self, filename):
        self.subject = filename
        docs = self.loader.load()

class PDFProcessor:
    def __init__(self, file_path, persist_directory='persist_chroma_pdf'):
        self.file_path = file_path
        self.persist_directory = persist_directory
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  # CLIP for both text and image embeddings
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")  # CLIP processor for images
        self.subject = None
        self.device= "cpu"

    def process(self, filename):
        """Process the PDF, extract text, images, and store them in Chroma."""
        self.subject = filename
        doc = fitz.open(self.file_path)
        output_image_dir = 'extracted_images'
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        text_embeddings = []
        Document=[]
        image_embeddings = []
        for page_idx in range(len(doc)):
            text, image_paths = self.extract_text_and_images_from_page(self.file_path, page_idx,output_image_dir)
            
            text_chunks = text.split("\n")  # Split text into chunks
            
            image_embeds_with_path = self.get_image_embeddings(image_paths) # getting the image embedding for all the images in the page
            
            # Get text embeddings for each chunk
            for text_chunk in text_chunks:
                text_embedding = self.get_text_embedding(text_chunk)
                Document.append(text_chunk)
                text_embeddings.append(text_embedding)
            
            # Store the tuple containing image embeddings and the path into the list for each page
            for img_embedding in image_embeds_with_path:
                image_embeddings.append(img_embedding)
        match_vector= self.Meta_data(text_embeddings,image_embeddings)
        text_embedding.tolist()
        collection=chroma_client.get_or_create_collection(name="PDF_Documents")
        collection.add(
            documents=Document,
            ids=[f"id{x}" for x in range(1,len(text_embeddings)+1)],
            embeddings=text_embeddings,
            metadatas=match_vector,
            
        )

 # the function return the text and the image path where the extracted images are saved   
    def extract_text_and_images_from_page(self,pdf_path, page_number,output_image_dir):
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number)
        text = page.get_text("text")
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)
        image_paths = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_data = base_image["image"]

            # Convert image data to a Pillow Image object
            image = Image.open(BytesIO(image_data))

            # Generate image filename based on page and image index
            image_filename = f"page_{page_number + 1}_img_{img_index + 1}.jpg"
            image_path = os.path.join(output_image_dir, image_filename)
            
            # Save the image to the specified directory
            if image.mode == 'RGBA':
            # Convert to RGB
                image = image.convert('RGB')
            image.save(image_path)
            image_paths.append(image_path)

        return text, image_paths

    def get_text_embedding(self,text):
        inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_embedding = clip_model.get_text_features(**inputs)
        return text_embedding.cpu().numpy()

    def get_image_embeddings(self, image_paths):
        image_embeddings_with_paths = []  # List to store (embedding, path) pairs
        for image_path in image_paths:
            # Open the image file from disk
            image = Image.open(image_path)

            # Preprocess the image and get embeddings
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_embedding = self.clip_model.get_image_features(**inputs)
            
            # Store the embedding along with the path as a tuple
            image_embeddings_with_paths.append((image_embedding.cpu().numpy(), image_path))

        return image_embeddings_with_paths

    def Meta_data(self,text_embeddings, image_paths_with_embeddings, similarity_threshold=0.5):
        # NOte experiment with the threshold for better result
        matched_results = []

        # Iterate through each text embedding
        for text_idx, text_emb in enumerate(text_embeddings):
            # Prepare a dictionary to store the matches for this text embedding
            matched_images = {'subject': self.subject, 'file_type': 'pdf', "image_paths": [],'image_embeddings':[]}
            
            # Iterate through image paths and embeddings
            for img_emb,img_path in image_paths_with_embeddings:
                # Calculate cosine similarity between the text embedding and the image embedding
                similarity = cosine_similarity(text_emb.reshape(1, -1), img_emb.reshape(1, -1))[0][0]
                
                # If the similarity is above the threshold, consider this image as a match
                if similarity >= similarity_threshold:
                    matched_images["image_paths"].append(img_path)
                    matched_images['image_embeddings'].append(img_emb)
            
            # If there are matching images for the current text embedding, add it to the results
            if matched_images["image_paths"]:
                matched_results.append(matched_images)
            else:
                matched_results.append({'subject': self.subject, 'file_type': 'pdf', "image_paths": [math.nan],'image_embeddings':[math.nan]})

        return matched_results

# Main processing function
def process_file(file_path):
    """Process a file (CSV, PDF, etc.) by determining its type and calling the appropriate processor."""
    file_extension = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)
    file = os.path.splitext(file_name)
    filename = file[0]
    
    if file_extension == '.csv':
        processor = CSVProcessor(file_path)
    elif file_extension in ['.pdf', '.txt', '.docx']:
        processor = PDFProcessor(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    print(filename)
    processor.process(filename)

    # Return the processed data and vector stores for further use
   # return extracted_docs, image_embeddings, image_paths, text_vectordb # the last line is not needed
