import streamlit as st
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import pinecone 
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import BedrockEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import AmazonKendraRetriever
from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore

import boto3
import toml


PINECONE_API_KEY = st.secrets.PINECONE_API_KEY
PINECONE_ENV = st.secrets.PINECONE_ENV
openai_api_key = st.secrets.OPENAI_API_KEY
kendra_index = st.secrets.KENDRA_INDEX
bedrock_region = st.secrets.AWS_BEDROCK_REGION
COHERE_API_KEY = st.secrets.COHERE_API_KEY
#os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
max_tokens = 1024  # Adjust as needed
temperature = 0.7  # Adjust as needed
index_pinecone_hsdemocracy  = 'unidosus-edai-hsdemocracy'
index_pinecone_asu  = 'unidosus-edai-asu'
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
embeddings = BedrockEmbeddings(client=bedrock_client, region_name="us-east-1")
text_field = "text"
index_pinecone = 'unidosus-edai-hsdemocracy'

# Setup bedrock


def embedding_db(index_name_param):
    # we use the openAI embedding model
    embeddings = BedrockEmbeddings(client=bedrock_client, region_name="us-east-1")
    index_name = index_name_param
    #strat the Pinecone Index
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    text_field = "text"
    index = pc.Index(index_name)
    vectorstore = PineconeVectorStore(index, embeddings, text_field)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 50})
    return retriever
   
# Function to retrieve answers
def retrieval_answer(query, llm_model, vector_store):        
    # Select the model based on user choice
    if llm_model == 'Anthropic Claude V3':
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"        
        llm = BedrockChat(model_id=model_id, streaming=True)
    elif llm_model == 'GPT-4-1106-preview':
        llm = ChatOpenAI(model_name="gpt-4-1106-preview",openai_api_key = openai_api_key)

    else:
        return "Invalid LLM model selection."
    
     # Select the Retriever based on user choice
    if vector_store == 'Pinecone: Highschool democracy':
        retriever = embedding_db(index_pinecone_hsdemocracy)
        response = retrieval_answer(query,llm,retriever)
    elif vector_store == 'Pinecone: University of Arizona':
        retriever = embedding_db(index_pinecone_asu)
        response = retrieval_answer(query,llm,retriever)
    return response

from operator import itemgetter
from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_core.prompts import ChatPromptTemplate

def format_docs(docs: List[Document]) -> str:
    """Convert Documents to a single string.:"""
    formatted = [
        f"Article Title: {doc.metadata['title']}\nArticle Snippet: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You're a helpful AI assistant. Given a user question answer the user question generating a comprensive analysis
                If none of the retrieved documents answer the question, just say you don't know.\n\nHere are the retrieved docuemnts from vector db:{context}",
            """      
         ),
        ("human", "{question}"),
    ]
)

def pinecone_db():
    """
    Initializes and returns the Pinecone index.
    """
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_pinecone)
    return index

def retrieval_answer(query,llm,retriever_2):
    index = pinecone_db()
    vectorstore = PineconeVectorStore(index, embeddings, text_field)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 100})
    compressor = CohereRerank(top_n = 20)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    format = itemgetter("docs") | RunnableLambda(format_docs)
    # subchain for generating an answer once we've done retrieval
    answer = prompt | llm | StrOutputParser()
    # complete chain that calls wiki -> formats docs to string -> runs answer subchain -> returns just the answer and retrieved docs.
    chain = (
        RunnableParallel(question=RunnablePassthrough(), docs=compression_retriever)
        .assign(context=format)
        .assign(answer=answer)
        .pick(["answer", "docs"])
    )
    respond = chain.invoke(query)
    return respond