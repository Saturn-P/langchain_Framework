"""
Simple doc-based QA pipeline:
- load text (local .txt or .md)
- create embeddings
- use a vectorstore (in-memory or FAISS) to retrieve relevant chunks
- run an LLM to answer

This is a minimal illustrative example; for larger docs use chunking and persistent vectorstores.
"""

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

def main():
    # Load a simple text document
    loader = TextLoader("examples/sample_doc.txt", encoding="utf8")
    docs = loader.load()

    # Create embeddings and vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Make a retriever and a RetrievalQA chain
    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), chain_type="stuff", retriever=retriever)

    question = "Summarize the main point of the document."
    answer = qa.run(question)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
