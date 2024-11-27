import streamlit as st
from RAG_Pipeline import initialize_RetrievalQA_pipeline  # Import your pipeline initialization function
import os
from dotenv import load_dotenv
import logging
import re



# Load environment variables
load_dotenv()

# Set up page configuration
st.set_page_config(
    page_title="Medical QA Bot",
    page_icon="🩺",
    layout="wide"
)

# Title
st.title("Medical QA Bot 🩺")
st.write("Ask any medical or clinical questions, and I'll try to provide an answer!")



#------------------------------------------------------------------------------------------------------#

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#-------------------------------------- PINECONE_API_KEY --------------------------------------# 
# Retrieve Pinecone API Key
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not pinecone_api_key:
    logger.error("Pinecone API Key is missing.")
    raise ValueError(
        "Please make sure to add your Pinecone API Key to the `.env` file.\n"
        "Find your Pinecone API Key here: https://app.pinecone.io/"
    )
else:
    logger.info("Pinecone API Key successfully loaded.")

#-------------------------------------- HUGGING_FACE_TOKEN --------------------------------------# 
# Retrieve the Hugging Face API token
hf_api_token = os.getenv("HF_HUB_API_TOKEN")

# Check if the token exists
if not hf_api_token:
    raise ValueError(
        "Hugging Face Hub API token is missing. "
        "Please add it to your `.env` file as HF_HUB_API_TOKEN."
    )

#-------------------------------------- PINECONE_INDEX_NAME --------------------------------------# 
# Retrieve the Pinecone Index Name
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

#------------------------------------------------------------------------------------------------------#





# Initialize the RAG pipeline
try:
    qa_pipeline = initialize_RetrievalQA_pipeline(pinecone_index_name, pinecone_api_key, hf_api_token,llm_model_id="meta-llama/Llama-3.2-3B-Instruct")
    st.success("Pipeline initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize the pipeline: {e}")
    st.error(f"Error initializing the pipeline: {e}")
    st.stop()


# Input text box for queries
query = st.text_input("Enter your medical question:", placeholder="E.g., What is paracetamol?")

# Checkbox to show source documents
show_sources = st.checkbox("Show source documents")

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a question to get an answer.")
    else:
        with st.spinner("Fetching answer..."):
            try:
                # Call the RAG pipeline with the query
                # result = qa_pipeline({"query": query})
                result = qa_pipeline({"query": query, "max_length": 512})  # Adjust as needed


                # Extract the helpful answer
                match = re.search(r"Helpful Answer:\s*(.*)", result['result'])
                helpful_answer = match.group(1).strip() if match else "No helpful answer found."
                st.write(f"**Answer:** {helpful_answer}")

                # Display source documents if the checkbox is checked
                if show_sources:
                    source_docs = result["source_documents"]
                    if source_docs:
                        st.write("### Source Documents")
                        for i, doc in enumerate(source_docs, start=1):
                            # Extract attributes directly from the Document object
                            st.markdown(f"### Document {i}")
                            st.markdown(f"**ID:** {doc.id}")  # Accessing the 'id' attribute
                            st.markdown(f"**Title:** {doc.metadata.get('title', 'No Title')}")  # Safely accessing 'metadata'
                            st.markdown(f"**Content:**\n{doc.page_content}")  # Accessing 'page_content'
                            st.markdown("---")  # Divider for each document
                    else:
                        st.info("No source documents available.")


            except Exception as e:
                # Log the error and show an error message
                logger.error(f"Failed to process the query: {e}")
                st.error(f"An error occurred while processing your query: {e}")
