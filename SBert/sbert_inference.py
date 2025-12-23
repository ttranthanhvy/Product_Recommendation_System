import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

try:
    import config
except ImportError:
    from SBert import config


class SBERTRecommender:
    def __init__(self):
        print("Khởi tạo SBERT Recommender...")
        self.load_data()

    def load_data(self):
        try:
            # Chỉ load vector số, cực nhanh và nhẹ RAM
            self.embeddings = np.load(config.VECTOR_SAVE_PATH)
            self.product_ids = np.load(config.ID_MAPPING_PATH, allow_pickle=True)

            # Tạo map index để tra cứu nhanh O(1)
            self.product_id_to_index = {pid: i for i, pid in enumerate(self.product_ids)}
            print("Đã load xong vector và ID mapping.")
        except FileNotFoundError:
            print("Không thấy file vector")
            self.embeddings = None

    def get_recommendations(self, target_product_id, top_k=10):
       
        if self.embeddings is None:
            return []

        if target_product_id not in self.product_id_to_index:
            print(f"Warning: Sản phẩm {target_product_id} không có trong kho dữ liệu vector.")
            return []

        idx = self.product_id_to_index[target_product_id]
        target_vector = self.embeddings[idx].reshape(1, -1)

        sim_scores = cosine_similarity(target_vector, self.embeddings)[0]

        sorted_indices = sim_scores.argsort()[::-1]
        top_indices = sorted_indices[1: top_k + 1]  

        results = []
        for i in top_indices:
            results.append({
                'product_id': self.product_ids[i],
                'sbert_similarity': float(sim_scores[i])  
            })

        return results


if __name__ == "__main__":
    recommender = SBERTRecommender()

    
    test_id = recommender.product_ids[0] if hasattr(recommender, 'product_ids') else "BOOK_001"
    print(f"\nGợi ý cho sản phẩm: {test_id}")

    items = recommender.get_recommendations(test_id, top_k=20)
    for item in items:
        print(item)