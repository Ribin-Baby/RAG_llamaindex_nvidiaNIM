from llama_index.core import Settings
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank

from fastapi import FastAPI

from dotenv import load_dotenv
import os

load_dotenv() 

app = FastAPI()


# In-memory database (for demonstration purposes)
chathistory: List = []

# Here we are using mixtral-8x7b-instruct-v0.1 model from API Catalog
Settings.llm = NVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1", nvidia_api_key=os.environ['NVIDIA_API_KEY'])
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", nvidia_api_key=os.environ['NVIDIA_API_KEY'], truncate="END")
Settings.text_splitter = SentenceSplitter(chunk_size=400)
DATA_PATH = f"{os.getcwd()}/data"
documents = SimpleDirectoryReader(DATA_PATH).load_data()
index = VectorStoreIndex.from_documents(documents)

reranker_query_engine = index.as_query_engine(
    similarity_top_k=40, node_postprocessors=[NVIDIARerank(top_n=4)]
)

@app.get("/")
async def reload():
    return {"response": "http://localhost/docs -> for how to use !"}


# Create an item
@app.get("/query")
async def chat(query):
    global reranker_query_engine
    
    response = reranker_query_engine.query(
    query
    )
    # chathistory.append(item)
    return {"query": query, "response": response.response, "documents": response.metadata}
