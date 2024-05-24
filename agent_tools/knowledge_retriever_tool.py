"""knowledge_retriever_tool.py
Custom HDTWin tool that reads rules from a knowledge base previously written to a vector store.

NOTE: first, run python agent_tools/knowledge_retriever_tool.py to
    write the vector store.

@author Gina Sprint
@date 5/22/24
"""
import os
import json
import pandas as pd

from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool


KNOWLEDGE_VECTOR_STORE_PATH = os.path.join("agent_tools", "vector_stores", "knowledge_vector_index")
# TEXT BASED RULES
# NOTE: these text markers are not included in the repository to protect participant privacy
JOURNAL_RULES = ["if journal_text is empty the more likely mild cognitive impairment",
                 "if journal_text has a large vocabulary, long sentences, and/or high sentence complexity then more likely healthy",
                 "if journal_text has a small vocabulary, short sentences, and/or low sentence complexity then more likely mild cognitive impairment",
                 "if journal_text uses positive emotion words then more likely healthy",
                 "if journal_text uses negative emotion words then more likely mild cognitive impairment",
                 "if journal_text appears to have more than one entry then more likely healthy",
                 "if journal_text appears to only have one entry then more likely mild cognitive impairment"
                 ]
INTERVIEW_ASSESSMENT_RULES = ["if some interview_assessment ratings are <= 3 then more likely mild cognitive impairment",
                             "if most interview_assessment ratings are >= 4 then more likely healthy",
                             "if the interview_assessment explanations suggest the participant confidently answered the questions correctly then more likely healthy",
                             "if the interview_assessment explanations suggest the participant struggled to answer the questions correctly then more likely mild cognitive impairment"
                             ]
rules_df = pd.read_csv(os.path.join("data", "dt_rules.csv"), index_col=0) 
RULES = [rule for rule in rules_df.index] + JOURNAL_RULES + INTERVIEW_ASSESSMENT_RULES

# MAKE DOCUMENT OBJECTS FOR RULES
def load_docs(rules):
    documents = []
    for rule in rules:
        doc = Document(page_content=rule, metadata={"source": "knowledge_base"})
        documents.append(doc)
    return documents

# STORE EMBEDDINGS IN VECTOR STORE
def embed_and_store_docs(documents, embeddings_model):
    print("Writing database to vector storage")
    db = FAISS.from_documents(documents, embeddings_model)
    db.save_local(KNOWLEDGE_VECTOR_STORE_PATH)

# CHECK VECTOR STORE BY QUERYING IT
def check_vectorestore(embeddings_model):
    db = FAISS.load_local(KNOWLEDGE_VECTOR_STORE_PATH, embeddings_model, allow_dangerous_deserialization=True)
    query = "What is a rule about the learning slope variable?"
    sim_docs = db.similarity_search(query)
    print(sim_docs[0].page_content)

# CREATE AND RETURN TOOL
def setup_knowledge_retriever_tool():
    # load knowledge base rules from vectorstore
    embeddings_model = OpenAIEmbeddings()
    db = FAISS.load_local(KNOWLEDGE_VECTOR_STORE_PATH, embeddings_model, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "knowledge_base",
        """Use this tool to search for general rules about differences between healthy and mild cognitive impairment groups."""
    )
    return retriever_tool

if __name__ == "__main__":
    # example run: python agent_tools/knowledge_retriever_tool.py
    with open("keys.json") as infile:
        key_dict = json.load(infile)
        os.environ["OPENAI_API_KEY"] = key_dict["OPENAI_API_KEY"]

    documents = load_docs(RULES)
    embeddings_model = OpenAIEmbeddings()
    embed_and_store_docs(documents, embeddings_model)
    check_vectorestore(embeddings_model)

    retriever_tool = setup_knowledge_retriever_tool()
    print(retriever_tool.run({"query": "shape_score_sd"}))