from transformers import BertTokenizer, BertModel
import faiss
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

documents = [
    "Ferrari Roma was the fastes car that was ever built till history",
    "the lamborgini aventador beat the ferrari roma in 2016 as the fast car",
    "the bugatti was the slowest car ever made"
    
]
question = "What is the slowest car "

def encode(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the embedding from the [CLS] token (first token)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Shape: (1, 768)

doc_embeddings = np.vstack([encode(doc) for doc in documents]) 
query_embedding = encode(question)  

index = faiss.IndexFlatL2(doc_embeddings.shape[1]) 
index.add(doc_embeddings) 

D, I = index.search(query_embedding, k=1) 
top_doc = documents[I[0][0]] 

print('-------------------------------------')
print("Most relevant document:", top_doc)

