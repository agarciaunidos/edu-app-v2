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
#os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
max_tokens = 1024  # Adjust as needed
temperature = 0.7  # Adjust as needed
index_pinecone_hsdemocracy  = 'unidosus-edai-hsdemocracy'
index_pinecone_asu  = 'unidosus-edai-asu'


# Setup bedrock
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

def embedding_db(index_name_param):
    # we use the openAI embedding model
    embeddings = BedrockEmbeddings(client=bedrock_client, region_name="us-east-1")
    index_name = index_name_param
    #strat the Pinecone Index
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    text_field = "text"
    index = pc.Index(index_name)
    vectorstore = PineconeVectorStore(index, embeddings, text_field)
    return vectorstore
   
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
        source = 'Pinecone'
    elif vector_store == 'Pinecone: University of Arizona':
        retriever = embedding_db(index_pinecone_asu)
        source = 'Pinecone'
    else:
        return "Invalid Vector DB selection."
    #llm = Bedrock(model_id=model_id, region_name=bedrock_region, client=bedrock_client, model_kwargs=model_kwargs)

    if source == 'Pinecone':
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever.as_retriever())
        response = qa(query)
    elif source == 'Kendra':
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        response = qa(query)
    return response['result']

