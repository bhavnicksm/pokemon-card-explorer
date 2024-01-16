import os
from tqdm import tqdm, trange

import openai

from datasets import load_dataset

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
QDRANT_API_KEY=os.environ['QDRANT_API_KEY']

openai.api_key = OPENAI_API_KEY

pcds = load_dataset("bhavnicksm/PokemonCardsPlus", split='train')

def get_emb_text(example):
    text = ""
    text += f"Name: {example['name']}\n"
    text += f"Set Name: {example['set_name']}\n"
    text += f"Pokemon Caption: {example['blip_caption']}\n"
    text += f"Card Description: {example['caption']}\n"
    text += f"Pokemon Description: {example['pokemon_intro']}\n"
    text += f"Pokedex Entry: {example['pokedex_text']}"
    return text

pcds = pcds.map(lambda example: {"text" : get_emb_text(example)})
pcds = pcds.map(lambda example : {"pokemon_image" : example['pokemon_image'] if example['pokemon_image'] != None else ''})
pcds = pcds.map(lambda example : {"pokemon_intro" : example['pokemon_intro'] if example['pokemon_intro'] != None else ''})
pcds = pcds.map(lambda example : {"pokedex_text" : example['pokedex_text'] if example['pokedex_text'] != None else ''})
pcds = pcds.map(lambda example : {"blip_caption" : example['blip_caption'] if example['blip_caption'] != None else ''})

pcds = pcds.add_column(name='_id', column=[id for id in range(len(pcds))])

# get the embeddings from OpenAI Embed
embeddings = []

for obj in tqdm(pcds):
    response = openai.Embedding.create(
        model="text-embedding-ada-002", 
        input=obj['text']
    )

    emb = response['data'][0]['embedding']
    embeddings.append(emb)

pcds = pcds.add_column("embedding", embeddings)

qdrant_client = QdrantClient(
    url="https://b8f17857-c547-4c58-8ba8-b479a15dacad.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key=QDRANT_API_KEY,
)

qdrant_client.recreate_collection(collection_name="pokemon-cards", 
                                  vectors_config=models.VectorParams(size=1536,                         #size of text-embedding-ada-002
                                                                     distance=models.Distance.COSINE    
                                                                     ))

# get the points for upload
points = []

for obj in tqdm(pcds):
    emb = obj['embedding']
    id = obj['_id']

    check = qdrant_client.retrieve(collection_name="pokemon-cards", ids=[id])
    if check != []:
        continue

    payload = { k:v for (k, v) in obj.items() if k not in [ 'embedding', '_id' ] }

    ps = PointStruct(id = id, vector=emb, payload=payload)
    points.append(ps)

# upload all the points to the dataset
for p in tqdm(points):
    qdrant_client.upsert(collection_name='pokemon-cards', wait=True, points=[p])

# check if all the points exist in the day
for obj in tqdm(pcds):
    emb = obj['embedding']
    id = obj['_id']

    check = qdrant_client.retrieve(collection_name="pokemon-cards", ids=[id])
    if check != []:
        continue
    else:
        print(id)