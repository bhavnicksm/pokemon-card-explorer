import os
import cohere
import pinecone 
from datasets import load_dataset

from torch.utils.data import DataLoader

COHERE_API_KEY = os.environ['COHERE_API_KEY']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']

co = cohere.Client(COHERE_API_KEY)  

pinecone.init(      
	api_key=PINECONE_API_KEY,      
	environment='us-west1-gcp'      
)      
index = pinecone.Index('pokemon-cards')



pcds = load_dataset("TheFusion21/PokemonCards", split='train')
pcds = pcds.map(lambda example:
                {"text" : f"Name: {example['name']}\nSet Name: {example['set_name']}\nDescription: {example['caption']}"})

dl = DataLoader(pcds, batch_size=64)

pinecone_obj = {'id': None, 'values': None, 'metadata' : None}

for batch in dl:
    # print(batch)
    upsert_list = []
    texts = batch['text']
    response = co.embed(texts=texts, model="embed-english-light-v2.0")
    embs = response.embeddings
    # print(embs[0])

    for i in range(len(batch['id'])):
        pcd = pinecone_obj.copy()
        pcd['id'] = batch['id'][i]
        pcd['values'] = embs[i]
        pcd['metadata'] = {"img_url": batch['image_url'][i],
                           "name" : batch['name'][i], 
                           "description": batch['caption'][i], 
                           "hp": int(batch['hp'][i])}
        upsert_list.append(pcd)
    
    # print(upsert_list[0])
    upsert_resp = index.upsert(vectors=upsert_list)
    # sleep(1)
    # break

print(index.describe_index_stats())