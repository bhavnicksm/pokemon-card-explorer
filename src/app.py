import os

import streamlit as st
import cohere 
import openai

from qdrant_client import QdrantClient


st.set_page_config(
    page_title="Pokemon Card Explorer",
    page_icon="🔍",
    layout="centered",
    menu_items={
        'Get Help': 'mailto:bhavnicksm@gmail.com',
        'Report a bug': "https://github.com/bhavnicksm/pokemon-card-explorer/issues",
        'About': "Pokemon Card Explorer lets you do super power semantic search over 13K Pokemon cards, to find the one you are looking for!"
    }
)

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
COHERE_API_KEY = os.environ['COHERE_API_KEY']
QDRANT_API_KEY = os.environ['QDRANT_API_KEY']

# def init_pinecone():
#     # find API key at app.pinecone.io
#     pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")
#     return pinecone.Index('pokemon-cards-v2')

def init_qdrant():
    # initialise the qdrant_client
    qdrant_client = QdrantClient(
        url = "https://b8f17857-c547-4c58-8ba8-b479a15dacad.us-east4-0.gcp.cloud.qdrant.io:6333", 
        api_key=QDRANT_API_KEY,
    )
    return qdrant_client
    
def init_reranker():
    return cohere.Client(COHERE_API_KEY) 


index = init_qdrant()
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
query = st.text_input(label="**Search:**", placeholder="Describe the Pokemon Card you are looking for...", )

if query != "":
    with st.spinner(text="Similarity Searching..."):
        #create the query embedding using OpenAI embedding
        resp = retriever.create(model="text-embedding-ada-002", input=query)
        qemb = resp['data'][0]['embedding']

        # Use the qdrant index to get top 6 results
        # qcs = index.query(qemb, top_k=6, include_metadata=True)
        qcs = index.search('pokemon-cards', query_vector=qemb, limit=12, with_payload=True)

        qtexts = [sp.payload['text'] for sp in qcs]
        rr_resp = reranker.rerank(model="rerank-english-v2.0", query=query, documents=qtexts, top_n=3)
        reranked_index = [rr_resp[i].index for i in range(3)]

        rr_qcs = [qcs[ind].payload for ind in reranked_index]
        
        urls = [x['card_image'] for x in rr_qcs]

    with st.spinner(text="Loading Cards"):
        card(urls)
    
    st.balloons()