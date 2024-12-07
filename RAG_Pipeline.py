import os
import logging
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import streamlit as st

from pinecone import Pinecone

@st.cache_resource
def initialize_RetrievalQA_pipeline(pinecone_index_name,pinecone_api_key,hf_api_token,llm_model_id, top_k=5, chain_type='stuff'):
    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)
    vector_store = PineconeVectorStore(index=index, embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))


    # Initialize retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    # Initialize Hugging Face model
    hf_llm = HuggingFaceHub(
        repo_id=llm_model_id,
        model_kwargs={
        "max_new_tokens": 512,
        "return_full_text":True,
        "device":'auto',
        "top_p": 0.15,
         # "top_k": 0,
        # "truncation": True,
        "do_sample": True,  
        "repetition_penalty":1.1
        },
        huggingfacehub_api_token=hf_api_token
    )

    # Create RetrievalQA pipeline
    qa_pipeline = RetrievalQA.from_chain_type(llm=hf_llm, chain_type=chain_type, retriever=retriever, return_source_documents=True)

    return qa_pipeline 