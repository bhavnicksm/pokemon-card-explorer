import os

import streamlit as st
import pinecone
import cohere 

COHERE_API_KEY = os.environ['COHERE_API_KEY']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']

def init_pinecone():
    # find API key at app.pinecone.io
    pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")
    return pinecone.Index('pokemon-cards')
    
def init_retriever():
    return cohere.Client(COHERE_API_KEY) 

index = init_pinecone()
retriever = init_retriever()


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
query = st.text_input(label="Search:", placeholder="Describe the Pokemon Card you are looking for...", )

if query != "":
    with st.spinner(text="Similarity Searching..."):

        # xq = retriever.encode([query]).tolist()
        resp = retriever.embed(texts=[query], model="embed-english-light-v2.0")
        qemb = resp.embeddings[0]

        xc = index.query(qemb, top_k=6, include_metadata=True)
        
        urls = [x['metadata']['img_url'] for x in xc['matches']]

    with st.spinner(text="Loading Cards"):
        card(urls)
    
    st.balloons()