import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import config


def generate_and_save_embeddings():

    try:
        df = pd.read_csv(config.DATA_PATH)
        print({df.shape})
    except FileNotFoundError:
        print(config.DATA_PATH)
        return

    text_cols = ['title', 'category', 'gender', 'price_segment']
    df[text_cols] = df[text_cols].fillna('')
    print("[1/2] Text Embeddings...")
    
    df['combined_text'] = df['title'].astype(str) + " " + \
                          df['category'].astype(str) + " " + \
                          df['gender'].astype(str) + " " + \
                          df['price_segment'].astype(str)

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    product_embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)

    np.save(config.VECTOR_SAVE_PATH, product_embeddings)
    np.save(config.ID_MAPPING_PATH, df['product_id'].values)

    print(f"Vector shape cuối cùng: {product_embeddings.shape}")

if __name__ == "__main__":
    generate_and_save_embeddings()