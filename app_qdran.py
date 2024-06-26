import os
import hashlib
import json
import streamlit as st
from langchain_qdrant import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import List, Dict
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA

# Charger les variables d'environnement
load_dotenv()

# Configurer la page de l'application
st.set_page_config(page_title="⚖️ JuriChat ⚖️")
st.title("JuriChat ⚖️")


def init_session_state():
    if "vstore" not in st.session_state:
        st.session_state.vstore = None
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Comment puis-je vous aider ?"}]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

init_session_state()

with st.sidebar:
    st.title('Historique des Chats')
    for i, chat in enumerate(st.session_state.chat_history):
        cols = st.columns([1, 0.1, 0.2, 0.1])
        with cols[0]:
            if st.button(f"{chat['name']}", key=f"load_{i}"):
                st.session_state.messages = chat["messages"]
                st.rerun()
        with cols[2]:
            if st.button("❌", key=f"delete_{i}"):
                st.session_state.chat_history.pop(i)
                st.rerun()
    if st.button("Nouveau Chat", key="new_chat"):
        print("this works")
        st.session_state.messages = [{"role": "assistant", "content": "Comment puis-je vous aider ?"}]
        st.session_state.chat_history.append(
            {"name": f"Chat {len(st.session_state.chat_history) + 1}", "messages": st.session_state.messages.copy()}
        )
        st.rerun()


def hash_file(filepath):
    with open(filepath, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def split_articles(text):
    articles = text.split('Article ')
    return [f'Article {article.strip()}' for article in articles if article.strip()]


def load_embeddings():
    data_dir = "data"
    json_path = "existing_hashes.json"  # Path to the JSON file

    all_texts = []
    existing_hashes = set()

    # Load existing hashes from JSON if it exists
    try:
        with open(json_path, "r") as json_file:
            existing_hashes = set(json.load(json_file))
    except (FileNotFoundError, json.JSONDecodeError):
        pass  # Ignore if JSON file doesn't exist or is invalid

    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            file_hash = hash_file(filepath)

            if file_hash not in existing_hashes:
                pdf_loader = PyPDFLoader(filepath)
                pages = pdf_loader.load()

                texts = [doc.page_content for doc in pages]
                articles = []

                for text in texts:
                    articles.extend(split_articles(text))
                split_text = articles  # Each chunk is a complete article

                embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
                embeddings_data = {
                    "filename": filename,
                    "texts": split_text,
                    "embeddings": embeddings.embed_documents(split_text),
                    "hash": file_hash,
                }
                all_texts.extend(split_text)

                # Add document data to vectordb and update existing_hashes
                url = "http://localhost:6333"
                qdrant = Qdrant.from_texts(
                    split_text,
                    embeddings,
                    url=url,
                    collection_name="jurichat"
                )
                existing_hashes.add(file_hash)

    # Save updated existing_hashes to JSON
    with open(json_path, "w") as json_file:
        json.dump(list(existing_hashes), json_file)

    return qdrant


# Function to format documents for display
def format_docs(docs):
    assert isinstance(docs, list), f"Expected list, got {type(docs)}"
    return "\n\n".join([doc for doc in docs])


# Function to generate response using LLM and vector store
def generate_response(qdrant, prompt_input):
    if qdrant is None:
        st.error("Le magasin vectoriel n'a pas été initialisé.")
        return "Erreur: Le magasin vectoriel n'a pas été initialisé."

    vstore = qdrant
    llm = ChatOpenAI(model_name="gpt-4", verbose=True, openai_api_key=os.getenv('OPENAI_API_KEY'))
    retriever = vstore.as_retriever(search_kwargs={"k": 5}, return_source_documents=True)

    # Define the output parser for QA tasks
    class QAOutput(BaseModel):
        answer: str = Field(..., description="The summary answer to the question")
        sources: str = Field(..., description="The sources used to generate the answer")

    output_parser = PydanticOutputParser(pydantic_object=QAOutput)

    # Define the prompt template for the LLM
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            """Voici des extraits du code pénal : {context}\n\n""Question : {question}\n\n""Répondez à la question en fournissant un résumé synthétique. ""Si tu ne trouves pas de réponse, ne répond pas et répond que l'information n'est pas contenue dans le code pénal. ""Liste les articles utilisés pour générer la réponse sous forme de bullet point."""""
        ),
    )
    

    qa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=retriever,
                                       chain_type_kwargs={'prompt': prompt_template}
                                       )

    response = qa({'query': prompt_input})
    return response["result"]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))

qdrant = Qdrant.from_existing_collection(
embedding=embeddings,
collection_name="jurichat",
url="http://localhost:6333"
)

# Main chat loop
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(qdrant, prompt) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
    
    # Sauvegarder l'historique des chats
    st.session_state.chat_history.append({
        "name": f"Chat {len(st.session_state.chat_history) + 1}",
        "messages": st.session_state.messages.copy()
    })

# Sauvegarder l'historique des chats si c'est un nouveau message
if len(st.session_state.chat_history) == 0 or st.session_state.chat_history[-1]["messages"] != st.session_state.messages:
    st.session_state.chat_history.append(
        {"name": f"Chat {len(st.session_state.chat_history) + 1}", "messages": st.session_state.messages.copy()}
    )