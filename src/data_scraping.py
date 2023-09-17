import pandas as pd
from time import time, sleep
from tqdm import tqdm, trange
import requests
from bs4 import BeautifulSoup

url = "https://pokemondb.net/pokedex/all"
r = requests.get(url)
soup = BeautifulSoup(r.content, 'html5lib')

data = []
table_body = soup.find("tbody")
rows = table_body.find_all('tr')
for row in rows:
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]
    data.append([ele for ele in cols if ele])


urls = []
base_url = f"https://pokemondb.net/pokedex/"
for a in tqdm(data):
  name = a[1]
  name = name.lower().replace(" ", '-')
  candidate_url = base_url + f"{name}"
  r = requests.get(candidate_url)
  if r.ok:
    urls.append(candidate_url)


def get_pokedex_entries(url):
  r = requests.get(url)
  if not r.ok:
    print("URL is not responding...")
    return -1
  soup = BeautifulSoup(r.content, 'html5lib')

  pokedex_entries = soup.find_all("td", {"class" : "cell-med-text"})
  pokedex_text = " ".join([entry.text for entry in pokedex_entries])

  return pokedex_text

def get_pokemon_name(url):
  r = requests.get(url)
  if not r.ok:
    print("URL is not responding...")
    return -1
  soup = BeautifulSoup(r.content, 'html5lib')
  name = soup.find("h1").text
  return name

def get_pokemon_intro(url):
  r = requests.get(url)
  if not r.ok:
    print("URL is not responding...")
    return -1

  soup = BeautifulSoup(r.content, 'html5lib')
  ps = soup.find_all("p")
  texts = [p.text for p in ps]
  i = texts.index("\n\n\n")
  return " ".join(texts[:i])

def get_pokemon_image(url, name):
  r = requests.get(url)
  soup = BeautifulSoup(r.content, 'html5lib')
  try:
    img_url = soup.find_all("img", {"alt":f"{name} artwork by Ken Sugimori"})[0]['src']
  except:
    try:
      img_url = soup.find_all("img", {"alt": f"{name}"})[0]['src']
    except:
      return -1

  return img_url


p_names = []
pd_text = []
p_intros = []
p_images = []

for url in tqdm(urls):
  name = get_pokemon_name(url)
  p_names.append(name)

  intro = get_pokemon_intro(url)
  p_intros.append(intro)

  img_url = get_pokemon_image(url, name)
  p_images.append(img_url)

  pokedex_entry = get_pokedex_entries(url)
  pd_text.append(pokedex_entry)

  sleep(1)


pd.DataFrame.from_dict({"name":p_names,
                        "intro_text":p_intros,
                        "img_url":p_images,
                        "pokedex_entry": pd_text})\
  .to_json("./pokemondb_data.jsonl", lines=True, orient='records')