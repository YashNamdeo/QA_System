import os 
import glob
os.environ["OPENAI_API_KEY"] = "your-key-here"

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def load_documents():
    document_paths = "./documents/*"
    documents = []

    for path in glob.glob(document_paths):
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)

        loaded_documents = loader.load()
        documents.extend(loaded_documents)
    
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":1})
    # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

    return qa

def main():
    st.title("Interactive Chatbot")
    st.markdown("Ask a question:")

    question = st.text_input("Question", key="question")
    index = load_documents()

    if st.button("Get Answer"):
        if question:
            result = index({"query": question})
            answer = result['result']
            st.markdown(f"**Answer:** {answer}")
        else:
            st.warning("Please enter a question.")

    if st.checkbox("Ask More Questions"):
        more_questions = True
        counter = 1
        while more_questions:
            new_question = st.text_input(f"Question {counter}", key=f"new_question_{counter}")
            if st.button("Get Answer", key=f"new_answer_{counter}"):
                if new_question:
                    result = index({"query": new_question})
                    answer = result['result']
                    st.markdown(f"**Answer:** {answer}")
                else:
                    st.warning("Please enter a question.")
            more_questions = st.checkbox("Ask Another Question", key=f"more questions {counter}")
            counter += 1

if __name__ == "__main__":
    main()