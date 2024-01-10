import streamlit as st
import time

st.set_page_config(
    page_title="Document GPT",
    page_icon=":memo:",
)

st.title("Document GPT")

st.markdown(
    """
    ## Document GPT
    # Use to ask questions about your documents
    """
)

file = st.file_uploader(
    "Upload a file",
    type=["pdf", "txt", "docx"],
)
