__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os

os.environ['KAGGLE_USERNAME'] = st.secrets["KAGGLE_USERNAME"]
os.environ['KAGGLE_KEY'] = st.secrets["KAGGLE_KEY"]
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]

from kaggle import api 
import os
import pandas as pd 

from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
import chromadb
import base64

chromadb.api.client.SharedSystemClient.clear_system_cache()

if not os.path.exists("./doc_dataset"):
    api.dataset_download_files('ivanhusarov/the-batch-articles-initial', path='doc_dataset', unzip=True)
if not os.path.exists("./chroma"):
    api.dataset_download_files('ivanhusarov/the-batch-rag-chroma', path='chroma', unzip=True)

df = pd.read_csv('./doc_dataset/articles_html.csv')
df.set_index('Url', inplace=True)



######## Utility ########

def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
        
def encode_image_from_bytes(image):
    """Getting the base64 string"""
    return base64.b64encode(image.getvalue()).decode("utf-8")


def image_summarize(img_base64, prompt):
    """Make image summary"""
    chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)

    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content


def generate_img_summaries(img_file):
    
    prompt = f"""You are an assistant tasked with generating a query summary for an input image. \
    This query will be embedded and used to retrieve the most relevant raw image from a database. \
    Provide a concise, retrieval-optimized description that captures the key visual details and context of the image. \
    """

    base64_image = encode_image_from_bytes(img_file)

    return image_summarize(base64_image, prompt)


def create_multi_modal_query(question=None, image=None):
    query_arr = []
    if question:
        query_arr.append(question)

    if image:
        query_arr.append(f"Consider the following context: {generate_img_summaries(image)}")
        
    return " . ".join(str(element) for element in query_arr)
############# Utility #############




model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {"normalize_embeddings": True}

embedding_function = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
import chromadb

persistent_client = chromadb.PersistentClient(path="chroma/all_docs_collection")
collection = persistent_client.get_or_create_collection("all_docs_collection")

vector_store = Chroma(
    client=persistent_client,
    collection_name="all_docs_collection",
    embedding_function=embedding_function,
)


llm = ChatOpenAI(model='gpt-4o', temperature=0)

retriever = vector_store.as_retriever(search_kwargs={"k": 10})

compressor = CohereRerank(model="rerank-english-v3.0")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

template = """Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':

### CONTEXT
{context}

### QUESTION
Question: {question}
"""
prompt = PromptTemplate(
    template=template, 
    input_variables=[
        'context', 
        'question',
    ]
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                       retriever=compression_retriever,     
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": prompt}
)

st.title("📄 The Batch articles RAG system")
st.write(
    "Ask anything about tech and we will try to help you!",
)
st.write(    "At least image or question should be filled"
)
question = st.text_area(
    "Now ask your question!",
    placeholder="Explain the misclassification in AI, mentioning the bias in ML models?",
)

image_uploaded = st.file_uploader("Choose an image")
if image_uploaded is not None:
    st.image(image_uploaded)


if st.button('Process   ->'):
    query = create_multi_modal_query(question=question, image=image_uploaded)
    answer = qa_chain.invoke({"query": query})
    doc_id = answer['source_documents'][0].metadata['doc_id']
    relevance_score = answer['source_documents'][0].metadata['relevance_score']
    image_found = answer["source_documents"][0].metadata["image_url"]

    st.title("Final query:")
    st.write(query)

    st.title("Answer:")
    st.write(answer['result'])
    if image_found is not "":
        st.image(f"./doc_dataset/images_clean/{image_found}")

    if doc_id is not None:
        st.title("\nOriginal document:")
        st.write(f"Relevance score: {round(relevance_score, 2)}")
        st.html((df.loc[doc_id])['Content'])

        img_folder_path = f"./doc_dataset/images_clean/{os.path.basename(os.path.normpath(doc_id))}/"
        for img_path in os.listdir(img_folder_path):
            st.image(img_folder_path + img_path)
    
        st.write('\n\nCheck the source link:')
        st.write(answer['source_documents'][0].metadata['doc_id'])
