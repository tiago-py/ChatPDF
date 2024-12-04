import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback



with st.sidebar:
    st.text("ChatPDF")
    st.markdown('''
      ## Sobre o projeto
      ## Desenvolvido com uma LLM:
      - [LangChain](https://www.langchain.com)
      
      ## Ferramentas utilizadas:
      - [Streamlit](https://streamlit.io)
      - [Modelo OpenAI](https://openai.com) LLM Model
    ''')

    add_vertical_space(5)
    st.write('Feito com carinho por [tiago-py](https://github.com/tiago-py)')


load_dotenv()
def main():
    st.markdown("## Chat PDF")
    pdf = st.file_uploader("Envie o seu PDF", type='pdf')
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl, rb") as f:
                VectorStore = pickle.load(f)
            st.write("Embeddings Loaded from the Disk")
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings) 
            with open(f"{store_name}.pkl, wb") as f:
                pickle.dump(VectorStore, f)

    query = st.text_input("Faça perguntas sobre o PDF enviado.")    
    st.write(query)
    if query:
        docs = VectorStore.similarity_search(query=query, k=3)
        
        llm = OpenAI()
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:

            response = chain.run(input_documents=docs, question=query)
            print(cb)
        st.write(response)
        
        
        #st.write(docs)
        

if __name__ == '__main__':
    main()