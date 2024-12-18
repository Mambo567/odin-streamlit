import os
import streamlit as st
from IPython.display import Markdown
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import CohereEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import tempfile
import time



# Aplicar estilo personalizado
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5e1b9; /* Fondo claro */
        font-family: Arial, sans-serif; /* Fuente limpia */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Cargar variables de entorno
load_dotenv()

# Inicializar claves API
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Título de la aplicación
st.title("ODIN | The one-eyed All-Father")
st.markdown("Odin is your gateway to knowledge. Share any PDF file, and he will use the power of LangChain and Gemini to deliver accurate, insightful answers tailored to your queries.")

st.image("Odin2.jpg", caption="He is the one-eyed All-Father, who sacrificed his eye in order to see everything that happens in the world.")

# Subir archivo PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file:
    # Guardar archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name  # Obtener la ruta temporal

    # Cargar y procesar el PDF
    loader = PyPDFLoader(temp_file_path)
    data_on_pdf = loader.load()

    # Dividir el documento en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(data_on_pdf)
    st.info(f"The file was divided in {len(splits)} pieces.")

    # Crear embeddings y procesarlos en un vectorstore temporal
    embeddings_model = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, user_agent="streamlit-app")

# Crear un directorio temporal para el vectorstore
    temp_dir = tempfile.mkdtemp()
    try:
        # Procesar embeddings en lotes pequeños
        batch_size = 10  # Número de chunks por lote
        all_embeddings = []  # Lista para almacenar los embeddings generados
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            embeddings = embeddings_model.embed_documents([doc.page_content for doc in batch])
            all_embeddings.extend(embeddings)  # Guardar embeddings generados
            time.sleep(1)  # Pausa para evitar superar el límite de tokens

        # Crear el vectorstore usando los embeddings generados
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings_model,
            persist_directory=temp_dir  # Directorio temporal para el vectorstore
        )
    finally:
        # Limpia manualmente el directorio temporal
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        st.success("Embeddings generated correctly.")

    # Configurar el modelo LLM (Gemini) y Retriever
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)
    retriever = vectorstore.as_retriever()

    # Configurar el prompt
    prompt = hub.pull("rlm/rag-prompt")

    # Definir RAG Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Hacer preguntas
    question = st.text_input("What knowledge do you seek, my child?:")
    if question:
        response = rag_chain.invoke(question)
        st.subheader("Odin says:")
        st.markdown(response)