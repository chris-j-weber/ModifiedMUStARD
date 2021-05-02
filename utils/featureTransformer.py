import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def stemming(f):
    # Stemming our data
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()

    f = f.apply(lambda x: x.split())
    f = f.apply(lambda x : ' '.join([ps.stem(word) for word in x]))
    return f

def remove_non_letters(df):
    # Replacing special symbols and digits in headline column
    df['sentence'] = df['sentence'].apply(lambda s: re.sub('[^a-zA-Z]', ' ', s))
    if 'context' in df.columns:
        df['context'] = df['context'].apply(lambda s: re.sub('[^a-zA-Z]', ' ', s))
    return df

def tfidfvect(features):
    features = list(stemming(features))

    tv = TfidfVectorizer(max_features = 5000)         # vectorizing the data with maximum of 5000 features
    features = tv.fit_transform(features)
    return features, tv

def bert(features, model_type):
    model = SentenceTransformer(model_type)
    sentence_embeddings = model.encode(features)
    return sentence_embeddings

def universalEmbedding(features, model_url):
    import tensorflow_hub as hub
    model = hub.load(model_url)
    def embed(input):
        return model(input)
    message_embeddings = embed(features)

    return pd.DataFrame(message_embeddings)