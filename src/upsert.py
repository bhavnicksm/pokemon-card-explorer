import os
import openai
import pinecone 
from datasets import load_dataset

from torch.utils.data import DataLoader

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']

openai.api_key = OPENAI_API_KEY

pinecone.init(      
	api_key=PINECONE_API_KEY,      
	environment='us-west1-gcp'      
)      
index = pinecone.Index('pokemon-cards-v2')



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

dl = DataLoader(pcds, batch_size=64)

pinecone_obj = {'id': None, 'values': None, 'metadata' : None}

for batch in dl:

    upsert_list = []
    texts = batch['text']

    response = openai.Embedding.create(
    model="text-embedding-ada-002",
    input=texts
    )
    embs = [x['embedding'] for x in response['data']]

    for i in range(len(batch['id'])):
        pcd = pinecone_obj.copy()
        pcd['id'] = batch['id'][i]
        pcd['values'] = embs[i]
        pcd['metadata'] = {"img_url": batch['card_image'][i],
                           "pokimg_url" : batch['pokemon_image'][i] if batch['pokemon_image'][i] != None else '', 
                           "name" : batch['name'][i], 
                           "description": batch['caption'][i],
                           "pokemon_intro": batch['pokemon_intro'][i] if batch['pokemon_intro'][i] != None else '', 
                           "pokedex_entry": batch['pokedex_text'][i] if batch['pokedex_text'][i] != None else '', 
                           "blip_caption" : batch['blip_caption'][i] if batch['blip_caption'][i] != None else '',  
                           "hp": int(batch['hp'][i])}
        upsert_list.append(pcd)

    upsert_resp = index.upsert(vectors=upsert_list)
    
print(index.describe_index_stats())