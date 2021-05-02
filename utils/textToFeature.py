from utils.datareader import *
from featureTransformer import *
import jsonlines

print('reading in data...')
mustard_path = "../data/sarcasm_data.json"

data = read_mustard(mustard_path)
data = remove_non_letters(data)



print('preprocessing data...')
# getting features and labels
mustard_features = data['sentence']


def write_to_jsonl(embeddings, name):
    with jsonlines.open('../data/new-' + name + '-output.jsonl', mode='w') as writer:
        writer.write(embeddings)


#stsb-bert-base
name = 'stsb-bert-base'
embeddings = bert(mustard_features, name).tolist()
write_to_jsonl(embeddings, name)
print(len(embeddings))


#distilbert-base-nli-stsb-mean-tokens
name = 'distilbert-base-nli-stsb-mean-tokens'
embeddings = bert(mustard_features, name).tolist()
write_to_jsonl(embeddings, name)
print(len(embeddings))

#universal-sentence-encoder
name = 'universal-sentence-embeddings'
embeddings = universalEmbedding(mustard_features, 'https://tfhub.dev/google/universal-sentence-encoder/4').numpy().tolist()
write_to_jsonl(embeddings, name)
print(len(embeddings))


