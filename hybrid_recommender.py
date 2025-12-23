import sys
import os
import torch
import numpy as np
import pandas as pd

#import
sys.path.append(os.path.abspath("Bert4rec"))

try:
    import recommend as bert_module
    print("Đã kết nối thành công với BERT4Rec!")
except ImportError:
    print("Lỗi: Không tìm thấy file 'recommend.py' trong thư mục 'Bert4rec'.")
    sys.exit()

from SBert.sbert_inference import SBERTRecommender

class BERT4RecWrapper:
    def __init__(self):
        self.model = bert_module.model
        self.df = bert_module.df
        self.item2idx = bert_module.item2idx
        self.idx2item = bert_module.idx2item
        self.event2idx = bert_module.event2idx
        self.build_user_input = bert_module.build_user_input

    def get_user_history_items(self, user_id, limit=10):  # Tăng limit lên 10
        # Lấy lịch sử, sắp xếp mới nhất lên đầu
        user_df = self.df[self.df["user_id"] == user_id].sort_values("timestamp", ascending=False)

        if user_df.empty:
            return []

        seen = set()
        history_items = []

        for pid in user_df["product_id"]:
            if pid not in seen:
                history_items.append(pid)
                seen.add(pid)
                if len(history_items) >= limit:
                    break

        return history_items

    @torch.no_grad()
    def predict_with_scores(self, user_id, top_k=50):
        # Gọi lại hàm build input từ file gốc
        items, events, seen_items = self.build_user_input(self.df, user_id)

        if items is None:
            return []

        # Chạy model (Forward pass)
        logits = self.model(items, events)  # Shape: (1, T, V)

        # Lấy điểm số tại vị trí cuối cùng (MASK)
        scores = logits[0, -1]

        # Lấy Top K (Lấy dư ra gấp 3 lần để lọc trùng)
        # Lấy cả indices (id) và values (điểm số)
        topk = torch.topk(scores, top_k * 3)
        topk_indices = topk.indices.tolist()
        topk_scores = topk.values.tolist()

        results = []
        for idx, score in zip(topk_indices, topk_scores):
            pid = self.idx2item.get(idx)

            # Logic lọc item đã xem & item rác
            if pid and pid not in seen_items:
                results.append({
                    'product_id': pid,
                    'bert_score': float(score)  # Điểm thô (Logits)
                })

            if len(results) >= top_k:
                break
        return results

class HybridRecommender:
    def __init__(self):
        self.sbert = SBERTRecommender()
        self.bert4rec = BERT4RecWrapper()
        #NEW
        columns = self.bert4rec.df.columns
        if 'age' not in columns or 'gender' not in columns:
            print("Không tìm thấy cột 'age' hoặc 'gender' trong data2.csv!")
            print("Không chạy được tính năng lọc nhân khẩu học!")
        else:
            print("Đã tìm thấy cột thông tin User (age, gender) trong Log.")


    def normalize_scores(self, item_list, score_key):
        #Chuẩn hóa điểm số về khoảng [0, 1] (Min-Max Scaling)
        if not item_list:
            return {}

        scores = [item[score_key] for item in item_list]
        min_s = min(scores)
        max_s = max(scores)

        normalized_dict = {}
        for item in item_list:
            if max_s == min_s:
                norm_val = 1.0  # Tránh chia cho 0
            else:
                norm_val = (item[score_key] - min_s) / (max_s - min_s)

            normalized_dict[item['product_id']] = norm_val

        return normalized_dict

    def recommend_cold_start(self, age=None, gender=None, top_k=10):
        print(f"Người dùng mới (Cold-start) | tuổi={age}, giới tính={gender}")

        # --- BƯỚC 1: LỌC TẬP NGƯỜI DÙNG TƯƠNG TỰ (USER-BASED) ---
        df_log = self.bert4rec.df.copy()

        # Tạo cột ảo nếu chưa có
        if 'rating' not in df_log.columns: df_log['rating'] = 0.0
        if 'sold_count' not in df_log.columns: df_log['sold_count'] = 0.0

        target_users = df_log[['user_id', 'age', 'gender']].drop_duplicates()

        if not target_users.empty:
            # Lọc người dùng cùng giới tính
            if gender:
                target_users = target_users[target_users['gender'].str.lower() == gender.lower()]

            # Lọc người dùng cùng độ tuổi (Biên độ +/- 5)
            if age:
                target_users = target_users[
                    (target_users['age'] >= age - 5) &
                    (target_users['age'] <= age + 5)
                    ]

            valid_user_ids = target_users['user_id'].tolist()
            if valid_user_ids:
                df_log = df_log[df_log['user_id'].isin(valid_user_ids)]

        # Fallback: Nếu lọc xong mà rỗng thì dùng lại toàn bộ data
        if df_log.empty:
            df_log = self.bert4rec.df.copy()

        # --- BƯỚC 2: TÍNH ĐIỂM POPULARITY ---
        popularity_log = df_log['product_id'].value_counts().reset_index()
        popularity_log.columns = ['product_id', 'interaction_count']

        max_inter = popularity_log['interaction_count'].max()
        popularity_log['log_score'] = popularity_log['interaction_count'] / max_inter if max_inter > 0 else 0.0

        # --- BƯỚC 3: LẤY THÔNG TIN SẢN PHẨM & LỌC GIỚI TÍNH SẢN PHẨM (QUAN TRỌNG) ---
        product_df_sorted = df_log.sort_values('timestamp', ascending=False)

        # [QUAN TRỌNG] Lấy thêm cột 'gender_product' và 'category'
        cols_to_get = ['product_id', 'rating', 'sold_count', 'category']
        if 'gender_product' in product_df_sorted.columns:
            cols_to_get.append('gender_product')

        product_df = (
            product_df_sorted[cols_to_get]
            .drop_duplicates(subset='product_id', keep='first')
        )

        # [FIX LỖI 1] LỌC SẢN PHẨM THEO GIỚI TÍNH (Product Filtering)
        if gender and 'gender_product' in product_df.columns:
            # Logic: Giữ lại sản phẩm (Cùng giới tính) HOẶC (Unisex)
            # Giả sử trong data, giới tính sản phẩm là: 'Nam', 'Nữ', 'Unisex'
            # Cần chuẩn hóa về lowercase để so sánh: 'nam', 'nữ', 'unisex'

            user_gender_lower = gender.lower()  # ví dụ: 'nam' hoặc 'nữ'

            # Hàm kiểm tra logic giới tính
            def is_valid_gender(prod_gender):
                if pd.isna(prod_gender): return True  # Giữ lại nếu không có data
                p_gen = str(prod_gender).lower()

                if 'unisex' in p_gen: return True
                if (user_gender_lower == 'male') and ('nam' in p_gen or 'male' in p_gen): return True
                if (user_gender_lower == 'female') and ( 'nữ' in p_gen or 'female' in p_gen): return True
                return False

            # Áp dụng bộ lọc
            product_df = product_df[product_df['gender_product'].apply(is_valid_gender)]

        # Tiếp tục tính toán như cũ
        product_df['rating'] = product_df['rating'].astype(float)
        product_df['sold_count'] = product_df['sold_count'].astype(float)
        product_df['rating_norm'] = product_df['rating'] / 5.0
        max_sold = product_df['sold_count'].max()
        product_df['sold_norm'] = product_df['sold_count'] / max_sold if max_sold > 0 else 0.0

        # Merge
        merged = popularity_log.merge(product_df, on='product_id', how='inner')

        merged['final_score'] = (
                0.3 * merged['rating_norm'] +
                0.2 * merged['sold_norm'] +
                0.5 * merged['log_score']
        )

        # Sắp xếp giảm dần
        merged = merged.sort_values('final_score', ascending=False)

        # --- BƯỚC 4: [FIX LỖI 2] DIVERSITY CAPPING (LỌC TRÙNG DANH MỤC) ---
        # Áp dụng logic giống hệt hàm recommend chính để tránh toàn quần áo
        diverse_results = []
        seen_categories = {}

        for _, row in merged.iterrows():
            cat = row['category']

            # Luật: Mỗi danh mục không quá 2 sản phẩm
            if seen_categories.get(cat, 0) >= 2:
                continue

            diverse_results.append({
                'product_id': row['product_id'],
                #'final_score': row['final_score']
                # 'details': {
                #     'rating': row['rating'],
                #     'sold_count': row['sold_count'],
                #     'popularity': row['log_score'],
                #     'category': cat  # Debug xem category là gì
                #}
            })

            seen_categories[cat] = seen_categories.get(cat, 0) + 1

            if len(diverse_results) >= top_k:
                break

        # Nếu lọc xong mà thiếu, lấy thêm bù vào (bỏ qua luật category)
        if len(diverse_results) < top_k:
            # Lấy những ID chưa có trong diverse_results
            current_ids = {item['product_id'] for item in diverse_results}
            for _, row in merged.iterrows():
                if row['product_id'] not in current_ids:
                    diverse_results.append({
                        'product_id': row['product_id'],
                        #'final_score': row['final_score'],
                        # 'details': {'rating': row['rating'], 'sold_count': row['sold_count'],
                        #             'popularity': row['log_score']}
                    })
                    if len(diverse_results) >= top_k:
                        break

        return diverse_results

    def recommend(self, user_id, top_k=10, alpha=0.5, age=None, gender=None):  # Alpha 0.5 để cân bằng
        if user_id not in self.bert4rec.df['user_id'].unique():
            return self.recommend_cold_start(age=age, gender=gender, top_k=top_k)

        # --- BƯỚC 1: Lấy dữ liệu từ BERT4Rec (Hành vi tương lai) ---
        bert_items = self.bert4rec.predict_with_scores(user_id, top_k=top_k * 3)
        bert_norm = self.normalize_scores(bert_items, 'bert_score')

        # --- BƯỚC 2: Lấy dữ liệu từ SBERT ---
        # ĐỔI HÀM GỌI Ở ĐÂY: limit=10 là con số vàng cho bài toán này
        history_items = self.bert4rec.get_user_history_items(user_id, limit=10)

        sbert_scores_accumulated = {}

        if history_items:
            # print(f"   (SBERT Context: {len(history_items)} items -> {history_items})")

            for item_id in history_items:
                # Lấy top-5 item tương tự theo SBERT
                recs = self.sbert.get_recommendations(item_id, top_k=5)

                for item in recs:
                    pid = item['product_id']
                    score = item['sbert_similarity']  # raw cosine similarity

                    # MAX POOLING qua các anchor
                    if pid in sbert_scores_accumulated:
                        sbert_scores_accumulated[pid] = max(
                            sbert_scores_accumulated[pid],
                            score
                        )
                    else:
                        sbert_scores_accumulated[pid] = score
        else:
            print("   (User mới -> Bỏ qua SBERT)")

        sbert_items = [
            {'product_id': pid, 'sbert_score': score}
            for pid, score in sbert_scores_accumulated.items()
        ]

        sbert_norm = self.normalize_scores(sbert_items, 'sbert_score')

        # --- BƯỚC 3: Tổng hợp (Weighted Sum) ---
        all_products = set(bert_norm.keys()) | set(sbert_norm.keys())
        final_scores = []
        for pid in all_products:
            score_b = bert_norm.get(pid, 0.0)
            score_s = sbert_norm.get(pid, 0.0)

            # Công thức Hybrid
            final_score = (alpha * score_b) + ((1 - alpha) * score_s)

            final_scores.append({
                'product_id': pid,
                'final_score': final_score,
                'details': {'bert': score_b, 'sbert': score_s}
            })

        # Sắp xếp giảm dần
        final_scores.sort(key=lambda x: x['final_score'], reverse=True)



        # LỌC TRÙNG DANH MỤC
        final_scores.sort(key=lambda x: x['final_score'], reverse=True)

        diverse_results = []
        seen_categories = {}  # Đếm số lượng sản phẩm mỗi loại

        # Lấy thông tin category từ self.bert4rec.df
        # Tạo map pid -> category cho nhanh
        product_cat_map = self.bert4rec.df.set_index('product_id')['category'].to_dict()

        for item in final_scores:
            pid = item['product_id']
            # Lấy category, nếu không có thì đặt là 'Unknown'
            cat = product_cat_map.get(pid, 'Unknown')

            # Kiểm tra: Nếu category này đã xuất hiện quá 2 lần trong list kết quả -> Bỏ qua (để nhường chỗ cho loại khác)
            if seen_categories.get(cat, 0) >= 2:
                continue

            diverse_results.append(item)
            seen_categories[cat] = seen_categories.get(cat, 0) + 1

            if len(diverse_results) >= top_k:
                break

        # Nếu lọc xong mà thiếu (không đủ top_k), thì lấy thêm từ danh sách gốc bù vào
        if len(diverse_results) < top_k:
            remaining = [i for i in final_scores if i not in diverse_results]
            diverse_results.extend(remaining[:top_k - len(diverse_results)])

        return diverse_results

        return final_scores[:top_k]

if __name__ == "__main__":
    # Khởi tạo
    hybrid = HybridRecommender()

    results = hybrid.recommend_cold_start(age=20, gender='female', top_k=20)
    for item in results:
        # Bạn cần nhìn xem cột nào đang cao bất thường?
        # Nếu cột SBERT toàn 0.9 mà cột BERT toàn 0.1 -> SBERT đang chiếm quyền kiểm soát.
        print(f"{item['product_id']:<15}")