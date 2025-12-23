import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, '../products.csv') # Hoặc file sạch của bạn

VECTOR_SAVE_PATH = os.path.join(BASE_DIR, '../results_sbert/sbert_vectors.npy')
ID_MAPPING_PATH = os.path.join(BASE_DIR, '../results_sbert/product_ids.npy')