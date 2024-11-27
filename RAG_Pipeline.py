import os
import logging
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import streamlit as st

from pinecone import Pinecone

# #------------------------------------------------------------------------------------------------------#
# # Load environment variables from .env file
# load_dotenv()

# # Logging configuration
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# #-------------------------------------- PINECONE_API_KEY --------------------------------------# 
# # Retrieve Pinecone API Key
# pinecone_api_key = os.getenv("PINECONE_API_KEY")

# if not pinecone_api_key:
#     logger.error("Pinecone API Key is missing.")
#     raise ValueError(
#         "Please make sure to add your Pinecone API Key to the `.env` file.\n"
#         "Find your Pinecone API Key here: https://app.pinecone.io/"
#     )
# else:
#     logger.info("Pinecone API Key successfully loaded.")

# #-------------------------------------- HUGGING_FACE_TOKEN --------------------------------------# 
# # Retrieve the Hugging Face API token
# hf_api_token = os.getenv("HF_HUB_API_TOKEN")

# # Check if the token exists
# if not hf_api_token:
#     raise ValueError(
#         "Hugging Face Hub API token is missing. "
#         "Please add it to your `.env` file as HF_HUB_API_TOKEN."
#     )

# #------------------------------------------------------------------------------------------------------#

# # Initialize Pinecone and Embeddings
# def get_vectore_store(pinecone_index_name):
#     # pinecone.init(api_key=pinecone_api_key, environment=environment)

#     pc = Pinecone(api_key=pinecone_api_key)
#     index = pc.Index(pinecone_index_name)
#     vector_store = PineconeVectorStore(index=index, embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

#     return vector_store


# # Configure the retriever
# def get_retriever(pinecone_index_name):
#     vector_store = get_vectore_store(pinecone_index_name)
#     return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})



# def create_qa_chain(pinecone_index_name):
#     retriever = get_retriever(pinecone_index_name)

#     hf_llm = hf_llm = HuggingFaceHub(
#     repo_id="meta-llama/Llama-3.2-1B-Instruct",  # Replace with your Arabic model
#     model_kwargs={
#         "max_length": 512,
#         "truncation": True,
#         "do_sample": False  # Deterministic generation
#     },
#     huggingfacehub_api_token=hf_api_token
#     )
#     return RetrievalQA.from_chain_type(llm=hf_llm, retriever=retriever, return_source_documents=True)




@st.cache_resource
def initialize_RetrievalQA_pipeline(pinecone_index_name,pinecone_api_key,hf_api_token,llm_model_id):
    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)
    vector_store = PineconeVectorStore(index=index, embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))


    # Initialize retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Initialize Hugging Face model
    hf_llm = HuggingFaceHub(
        repo_id=llm_model_id,
        model_kwargs={"max_length": 1200, "truncation": True, "do_sample": False},
        huggingfacehub_api_token=hf_api_token
    )

    # Create RetrievalQA pipeline
    qa_pipeline = RetrievalQA.from_chain_type(llm=hf_llm, retriever=retriever, return_source_documents=True)

    return qa_pipeline