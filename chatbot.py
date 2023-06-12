import os 
os.environ["OPENAI_API_KEY"] = "your-key-here"

import streamlit as st
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

def load_documents():
    loader = PyPDFLoader("mydoc.pdf")
    documents = loader.load()

    index = VectorstoreIndexCreator(
        text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0),
        embedding=OpenAIEmbeddings(),
        vectorstore_cls=Chroma
    ).from_loaders([loader])

    return index

def main():
    st.title("Interactive Chatbot")
    st.markdown("Ask a question:")

    question = st.text_input("Question", key="question")
    index = load_documents()

    if st.button("Get Answer"):
        if question:
            answer = index.query(llm=OpenAI(), question=question, chain_type="stuff")
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
                    answer = index.query(llm=OpenAI(), question=new_question, chain_type="stuff")
                    st.markdown(f"**Answer:** {answer}")
                else:
                    st.warning("Please enter a question.")
            more_questions = st.checkbox("Ask Another Question", key=f"more questions {counter}")
            counter += 1

if __name__ == "__main__":
    main()
