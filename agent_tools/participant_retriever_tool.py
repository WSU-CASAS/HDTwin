"""participant_retriever_tool.py
Custom HDTWin tool that reads information about participants previously
    written to a vector store.

NOTE: first, run python agent_tools/participant_retriever_tool.py directly to
    write the vector store.

@author Gina Sprint
@date 5/22/24
"""
import os
import json

import pandas as pd

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool


PARTICIPANT_VECTOR_STORE_PATH = os.path.join("agent_tools", "vector_stores", "participant_vector_index")

# MAKE DOCUMENT OBJECTS FOR CSV DATA
def load_file_docs(file_path):
    loader = CSVLoader(file_path=file_path)
    file_docs = loader.load()
    return file_docs

def load_docs_without_diagnosis(file_paths):
    documents = []
    for file_path in file_paths:
        full_file_path = os.path.join("data", file_path)
        # if test file(s) have diagnosis in it, write a temp file
        # with diagnosis removed so it doesn't get into vector store
        df = pd.read_csv(full_file_path, index_col=0)
        if "diagnosis" in df.columns:
            df.pop("diagnosis")
            temp_file_path = os.path.join("data", "temp.csv")
            df.to_csv(temp_file_path)
        else:
            temp_file_path = full_file_path
        file_docs = load_file_docs(temp_file_path)
        documents.extend(file_docs)
        # delete temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    return documents

# STORE EMBEDDINGS IN VECTOR STORE
def embed_and_store_docs(documents, embeddings_model):
    print("Writing database to vector storage")
    db = FAISS.from_documents(documents, embeddings_model)
    # save to disk
    db.save_local(PARTICIPANT_VECTOR_STORE_PATH)

# CHECK VECTOR STORE BY QUERYING IT
def check_vectorestore(embeddings_model):
    db = FAISS.load_local(PARTICIPANT_VECTOR_STORE_PATH, embeddings_model, allow_dangerous_deserialization=True)
    query = "What is Sloan's shape_score_sd value?"
    sim_docs = db.similarity_search(query)
    print(sim_docs[0].page_content)

# CREATE AND RETURN TOOL
def setup_participant_retriever_tool():
    # load historical data from the vector store
    embeddings_model = OpenAIEmbeddings()
    db = FAISS.load_local(PARTICIPANT_VECTOR_STORE_PATH, embeddings_model, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "person_information",
        """Use this tool when you are given a specific person's name and you need to search for their information."""
    )
    return retriever_tool

if __name__ == "__main__":
    # example run: python agent_tools/participant_retriever_tool.py
    with open("keys.json") as infile:
        key_dict = json.load(infile)
        os.environ["OPENAI_API_KEY"] = key_dict["OPENAI_API_KEY"]

    file_paths = ["test_synthetic.csv"]
    documents = load_docs_without_diagnosis(file_paths)
    embeddings_model = OpenAIEmbeddings()
    embed_and_store_docs(documents, embeddings_model)
    check_vectorestore(embeddings_model)
    
    retriever_tool = setup_participant_retriever_tool()
    print(retriever_tool.run({"query": "Sloan"}))