from typing import Any, Dict, List, Optional
from uuid import UUID
import streamlit as st
import os
import toml
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, AzureOpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

secrets = toml.load(".streamlit/secrets.toml")
os.environ["AZURE_OPENAI_API_KEY"] = secrets["AZURE_OPENAI_KEY"]
os.environ["AZURE_OPENAI_ENDPOINT"] = secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_DEPLOYMENT_NAME = secrets["AZURE_DEPLOYMENT_NAME"]

st.set_page_config(
    page_title="Document GPT",
    page_icon=":memo:",
)


class ChatCallBackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    openai_api_version="2023-09-01-preview",
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallBackHandler()],
)


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_version="2023-09-01-preview",
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    retriever = vectorstore.as_retriever()

    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"role": role, "message": message})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(documents.page_content for documents in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Answer the question using ONLY the following context. If you don't know the answer, 
            just say you don't know. DO NOT make up an answer.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("Document GPT")

st.markdown(
    """
    Ask questions about your documents
    
    Upload documents to start
    """
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a file",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("Good to go", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask a question")

    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)


else:
    st.session_state["messages"] = []
