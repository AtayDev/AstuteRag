# Step 1: Import necessary libraries
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS  # Example vectorstore
from langchain.llms import OpenAI

# Step 2: Load your documents into a vectorstore
vectorstore = FAISS.load_local("path_to_vectorstore")  # Load pre-indexed documents

# Step 3: Initialize your language model
llm = OpenAI(model="gpt-4")

# Step 4: Create the Retrieval-based QA Chain
def create_retrieval_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain

# Step 5: Create the QA system
qa_system = create_retrieval_chain(vectorstore, llm)

# Step 6: Ask a question
question = "What are the key benefits of renewable energy?"
response = qa_system({"query": question})

# Step 7: Print the results
print("Answer:", response["result"])
print("Source Documents:", response["source_documents"])
