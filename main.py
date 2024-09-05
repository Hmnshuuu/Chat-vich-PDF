#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install streamlit')


# In[2]:


import streamlit as st


# In[3]:


get_ipython().system('pip install -r requirement.txt')


# In[4]:


from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


# In[5]:


# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Directly set the API key
# os.environ["GOOGLE_API_KEY"] = "AIzaSyBNRc2lCiB2ZXf8rGMDHfUIg43fNO5UZe4"

# # Retrieve the API key
# api_key = os.getenv("GOOGLE_API_KEY")
# AIzaSyBNRc2lCiB2ZXf8rGMDHfUIg43fNO5UZe4
get_ipython().system('pip install python-dotenv')


# In[6]:


GOOGLE_API_KEY="AIzaSyBNRc2lCiB2ZXf8rGMDHfUIg43fNO5UZe4"


# In[7]:


from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
api_key = os.getenv("GOOGLE_API_KEY")

# Configure the API client with the API key
# import ge


# In[8]:


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text   


# In[9]:


def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks 


# In[10]:


def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss-index")


# In[11]:


def get_conversational_chain():
    prompt_template="""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:

    """
    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)

    return chain


# In[12]:


def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db=FAISS.load_local("faiss-index",embeddings)
    docs=new_db.similarity_search(user_question)
    
    chain=get_conversational_chain()

    response=chain(
        {"input_documents":docs,"question":user_question},return_only_outputs=True
    )

    print(response)

    st.write("Reply: ",response["output_text"])
    


# In[13]:


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini")

    user_question=st.text_input("Ask a question from pdf files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs=st.file_uploader("Upload your PDF files and click on submit & process button",accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text=get_pdf_text(pdf_docs)
                text_chunks=get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


# In[14]:


# if __name__=="_main_":
main()


# In[15]:


# %system streamlit run main.ipynb
get_ipython().run_line_magic('system', '')


# In[ ]:


import subprocess

# Run Streamlit using subprocess
subprocess.run(['streamlit', 'run', 'main.py'])  # Assuming your main Streamlit app is a Python file


# In[ ]:


get_ipython().system('jupyter nbconvert --to script main.ipynb')


# In[ ]:


get_ipython().run_line_magic('system', 'streamlit run main.py')


# In[ ]:




