# Use Ollama to generate local embeddings
from langchain_ollama import OllamaEmbeddings
# Chroma vector DB integration for LangChain
from langchain_chroma import Chroma
# Represents a chunk of text with metadata
from langchain_core.documents import Document
# For checking if database directory exists
import os
# To read the review CSV file
import pandas as pd

# Load a CSV file that contains realistic restaurant reviews
df = pd.read_csv("realistic_restaurant_reviews.csv")

# Initialize the embedding model using Ollama (local model)
# Ensure the model is pulled with: ollama pull mxbai-embed-large
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Define the folder where Chroma will store its database files
db_location = "./chrome_langchain_db"

# If the folder doesn't exist, we need to add documents
db_exists = os.path.exists(db_location)

# Create a Chroma vector store instance
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Only run this if database does not yet exist
if not db_exists:
    documents = [] # Holds Document objects
    ids = []       # Holds unique IDs for each document
    for i, row in df.iterrows():
        unique_id = str(i) # Use row index as a string ID

        # Create a document that includes title + review text
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={
                "rating" : row["Rating"],
                "date" : row["Date"]
            },
            id=unique_id
        )
        ids.append(unique_id)
        documents.append(document)

    # Store documents in the vector store (Chroma will embed and save them)
    vector_store.add_documents(documents=documents, ids=ids)

# Convert the vector store into a retriever interface
# This allows you to ask a question and get the top 5 most relevant documents
retriever = vector_store.as_retriever(search_kwargs={"k" : 5})
# retriever = vector_store.as_retriever()

