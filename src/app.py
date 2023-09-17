import os

import streamlit as st
import pinecone
import cohere 
import openai


OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
COHERE_API_KEY = os.environ['COHERE_API_KEY']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']

def init_pinecone():
    # find API key at app.pinecone.io
    pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")
    return pinecone.Index('pokemon-cards-v2')
    
def init_reranker():
    return cohere.Client(COHERE_API_KEY) 


index = init_pinecone()
reranker = init_reranker()
retriever = openai.Embedding

def card(urls):
    figures = [f"""
        <figure style="margin-top: 5px; margin-bottom: 5px; !important;">
            <img src="{url}" style="width: 225px; padding-left: 5px; padding-right: 5px" >
        </figure>
    """ for url in urls]
    return st.markdown(f"""
        <div style="display: flex; flex-flow: row wrap; text-align: center; justify-content: center;">
        {''.join(figures)}
        </div>
    """, unsafe_allow_html=True)


st.image('assets/logo.png')
# st.markdown("<h1 style='text-align: center;'>Pokemon Card Explorer</h1>", unsafe_allow_html=True)
query = st.text_input(label="**Search:**", placeholder="Describe the Pokemon Card you are looking for...", )

if query != "":
    with st.spinner(text="Similarity Searching..."):
        #create the query embedding using OpenAI embedding
        resp = retriever.create(model="text-embedding-ada-002", input=query)
        qemb = resp['data'][0]['embedding']

        # Use the pinecone index to get top 6 results
        qcs = index.query(qemb, top_k=6, include_metadata=True)

        docs = [qc['metadata']['description'] for qc in qcs['matches']]
        rr_resp = reranker.rerank(model="rerank-english-v2.0", query=query, documents=docs, top_n=6)
        reranked_index = [rr_resp[i].index for i in range(len(docs))]

        rr_qcs = [qcs['matches'][ind] for ind in reranked_index]
        
        urls = [x['metadata']['img_url'] for x in rr_qcs]

    with st.spinner(text="Loading Cards"):
        card(urls)
    
    st.balloons()