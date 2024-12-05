import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API key not found in .env file or environment variables.")

# Step 1: Load PDF Documents
def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load_and_split())
    return documents

# Step 2: Embed Documents and Create a VectorStore
def create_vectorstore(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

# Step 3: Adaptive Internal Knowledge Generation
def generate_internal_knowledge(query, llm, max_passages=3):
    prompt = f"Generate reliable passages about the query:\nQuestion: {query}\n"
    response = llm.generate([prompt], max_length=150)
    return [{"source": "internal", "text": r} for r in response.generations[0].text]

# Step 4: Create a Retrieval-based QA Chain
def create_retrieval_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain

# Step 5: Astute RAG Implementation
def astute_rag(query, vectorstore, llm):
    # Step 1: Retrieve External Passages
    qa_chain = create_retrieval_chain(vectorstore, llm)
    retrieval_results = qa_chain.run({"query": query})
    external_passages = [{"source": "external", "text": doc.page_content} for doc in retrieval_results["source_documents"]]

    # Step 2: Generate Internal Knowledge
    internal_passages = generate_internal_knowledge(query, llm)

    # Step 3: Consolidate Knowledge
    all_passages = internal_passages + external_passages
    sources = [p["source"] for p in all_passages]
    texts = [p["text"] for p in all_passages]
    
    consolidation_prompt = (
        "Consolidate the following passages:\n"
        + "\n".join([f"({source}) {text}" for source, text in zip(sources, texts)])
        + f"\n\nQuestion: {query}\n"
        "Return consistent information, separate conflicting details, and exclude irrelevant information."
    )
    consolidated_response = llm.generate([consolidation_prompt], max_length=300).generations[0].text

    # Step 4: Finalize Answer
    final_prompt = (
        f"Based on the consolidated information below, answer the question:\n"
        f"Question: {query}\nConsolidated Information: {consolidated_response}\n"
        "Provide the most reliable answer with confidence."
    )
    final_answer = llm.generate([final_prompt], max_length=100).generations[0].text

    return final_answer

# Main Function
def main():
    # Step 1: Load PDF documents
    folder_path = "internal_data"
    documents = load_documents_from_folder(folder_path)

    # Step 2: Create VectorStore
    vectorstore = create_vectorstore(documents)

    # Step 3: Initialize LLM
    llm = OpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=openai_api_key
    )

    # Example Query
    query = "What are the effects of climate change on agriculture?"
    result = astute_rag(query, vectorstore, llm)

    print("Final Answer:", result)

if __name__ == "__main__":
    main()
