import json
import time
from tqdm import tqdm
from flair.embeddings import OneHotEmbeddings, DocumentPoolEmbeddings
from flair.datasets import UD_ENGLISH
from flair.data import Sentence

from elasticsearch import Elasticsearch, helpers

# load a corpus
corpus = UD_ENGLISH()

embeddings = OneHotEmbeddings.from_corpus(corpus, embedding_length=400)
document_embeddings = DocumentPoolEmbeddings([embeddings])

with open('data/future_corpus.json', 'r') as f:
   future_corpus = json.load(f)

client = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "eMMWUyLnKD8i6X-B-5K",), verify_certs=False
)

print(client.ping())

client.indices.delete(index="vectors")

client.indices.create(
    index="vectors",
    mappings={
        "properties": {
            "embeddings": {
                "type": "dense_vector",
                "dims": 400,
            },
        }
    },
)

flair_sentence = [Sentence(s) for s in tqdm(future_corpus[:5000]) if s]

start_time = time.time()
for i in tqdm(range(0, len(flair_sentence), 1000)):
    try:
        document_embeddings.embed(flair_sentence[i:i+1000])
        list_of_docs = [{'embeddings': s.embedding.detach().numpy(), "_index": "vectors"} for s in flair_sentence[i:i+1000]]
        helpers.bulk(client, list_of_docs)
    except Exception as e:
        print(e)

client.indices.refresh(index="vectors")
print(time.time()- start_time)
