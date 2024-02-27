import os
import streamlit as st
from langchain.llms import Ollama
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.vector_stores import DeepLakeVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.service_context import ServiceContext

llm = Ollama(model="solar")
my_activeloop_org_id = "tobeetaylor"
my_activeloop_dataset_name = "dias-dev-2"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=False)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-base-en-v1.5")

title = "DIAS: Mit Daten chatten"
st.set_page_config(page_title=title, page_icon="🦙", layout="wide")

st.title(f"DIAS 💬")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Stelle mir Fragen über LEAM!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Dokumente werden geladen. Bitte warten Sie! Dies sollte 1-2 Minuten dauern."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()

        index = VectorStoreIndex.from_documents(docs, service_context=service_context, storage_context=storage_context)

        return index

index = load_data()

if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Frage stellen"):
    st.session_state.messages.append({"role": "user", "content": f"{prompt}."})



for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Antwort wird gesucht..."):
            response = st.session_state.chat_engine.chat(f"{prompt}")
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)


### Sidebar
emoji_map = {
    'txt': '📄',
    'pdf': '📄',
    'mp3': '🎵',
    'wav': '🎵',
    'mp4': '🎞️',
    'avi': '🎞️',
    'jpg': '🖼️',
    'png': '🖼️',
    'zip': '🗂️',
    'rar': '🗂️',
}

data_dir = './data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

existing_files = os.listdir(data_dir)

# Sidebar-Bereich für die Anzeige der vorhandenen Dateien
st.sidebar.write("Vorhandene Dateien")

# Datei-Explorer-Layout mit Emojis
for file_name in existing_files:
    file_extension = file_name.split('.')[-1]
    file_emoji = emoji_map.get(file_extension, '📁')
    st.sidebar.text(f"{file_emoji} {file_name}")

# Datei-Uploader in der Sidebar
uploaded_files = st.sidebar.file_uploader("", accept_multiple_files=True, label_visibility='hidden')

if uploaded_files:
    files_uploaded = False
    for uploaded_file in uploaded_files:
        file_path = os.path.join(data_dir, uploaded_file.name)
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.sidebar.success(f"Datei '{uploaded_file.name}' erfolgreich hochgeladen!")
            files_uploaded = True
        else:
            st.sidebar.error(f"Datei '{uploaded_file.name}' existiert bereits!")
    
    if files_uploaded:
        index = load_data()