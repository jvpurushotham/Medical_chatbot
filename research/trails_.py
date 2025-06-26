from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # updated import
import os

from sympy.physics.units import temperature


def load_pdf_file(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

extract_data = load_pdf_file(data_path=os.path.abspath('../Data/'))

def text_split(extract_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extract_data)
    return text_chunks

text_chunks = text_split(extract_data)
print("Length of text_chunks:", len(text_chunks))

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-V2')  # fixed param
    return embeddings

embeddings = download_hugging_face_embeddings()

query_vector = embeddings.embed_query("Hello world")
print("Embedding vector length:", len(query_vector))
print("First 5 dimensions of the embedding:", query_vector[:5])

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")

pc = Pinecone(api_key=PINECONE_API_KEY)

import os
index_name = 'testbot'

if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    print(f"Index '{index_name}' created.")
else:
    print(f"Index '{index_name}' already exists.")


import os
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks,
    index_name = index_name,
    embedding = embeddings,
)

from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embeddings,
)

retriever = docsearch.as_retriever(search_type = 'similarity', search_kwargs = {"k":3})
retrieved_docs = retriever.invoke("What is Acne?")

from langchain_community.llms import OpenAI
llm = OpenAI(temperature = 0.4, max_tokens = 500)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

sys_prompt = (
    "You are an assistant for answering the tasks."
    "Use the following pieces of retrieved context to answer"
    "the question. If you don't know the answer, say that you"
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", sys_prompt),
        ("human", f"{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

res = rag_chain.invoke({"input": "What is Acne?"})
print(res["answer"])

