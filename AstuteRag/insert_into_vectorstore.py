from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader

# Step 1: Load and extract text from the PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Step 2: Split text into chunks
def split_text_into_chunks(text, chunk_size=500, overlap=100):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

# Step 3: Convert text chunks to embeddings
def create_embeddings(text_chunks, embedding_model="all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectors = [embeddings.embed_query(chunk) for chunk in text_chunks]
    return embeddings, vectors

# Step 4: Insert embeddings into a vectorstore
def insert_into_vectorstore(text_chunks, embeddings, vectorstore_path="vectorstore"):
    vectorstore = FAISS(embedding_function=embeddings)
    vectorstore.add_texts(texts=text_chunks)
    vectorstore.save_local(vectorstore_path)
    return vectorstore

# Main function to process a PDF
def process_pdf_to_vectorstore(pdf_path, vectorstore_path="vectorstore"):
    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(text)} characters of text.")
    
    # Split text into manageable chunks
    text_chunks = split_text_into_chunks(text)
    print(f"Split text into {len(text_chunks)} chunks.")
    
    # Create embeddings
    embeddings, _ = create_embeddings(text_chunks)
    
    # Insert into vectorstore
    vectorstore = insert_into_vectorstore(text_chunks, embeddings, vectorstore_path)
    print(f"Saved vectorstore to {vectorstore_path}.")
    return vectorstore

# Run the process for a sample PDF
vectorstore = process_pdf_to_vectorstore("internal_data\Easy_recipes.pdf")
