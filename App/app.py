import streamlit as st
import pandas as pd
from datetime import datetime
import os
import sys

def load_css(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("App\style.css")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
from hybrid_recommender import HybridRecommender

st.set_page_config(page_title="Hybrid Recommender", layout="wide")
DATA_PATH = "clean_data.csv"

@st.cache_data
def load_log():
    return pd.read_csv(DATA_PATH)

df_log = load_log()

USERS_PATH = "App\\users.csv"

@st.cache_data
def load_users():
    if os.path.exists(USERS_PATH):
        return pd.read_csv(USERS_PATH)
    return pd.DataFrame(columns=[
        "user_id", "name", "birth_year", "age", "gender", "created_at"
    ])

df_users = load_users()


# LOAD MODEL 
@st.cache_resource
def load_hybrid():
    return HybridRecommender()

hybrid = load_hybrid()


if "page" not in st.session_state:
    st.session_state.page = "login"

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "user_info" not in st.session_state:
    st.session_state.user_info = {}

if "new_user_id" not in st.session_state:
    st.session_state.new_user_id = None


def calculate_age(birth_year):
    return datetime.now().year - birth_year

def render_products(results):
    product_ids = [r['product_id'] for r in results]

    product_df = (
        df_log[df_log['product_id'].isin(product_ids)]
        .sort_values("timestamp", ascending=False)
        .drop_duplicates("product_id", keep="first")
        .head(20)   # 4 x 5
    )

    cards_html = ""
    for _, row in product_df.iterrows():
        cards_html += f"""
        <div class="card">
            <div class="image-container">
                <img src="{row['image_url']}">
            </div>
            <h4>{row['title']}</h4>
            <p class="price">{int(row['price']):,} đ</p>
        </div>
        """

    st.markdown(
        f"""
        <div class="product-grid">
            {cards_html}
        </div>
        """,
        unsafe_allow_html=True
    )
def generate_new_user_id(existing_ids): 
    existing_ids = set(map(str, existing_ids)) 
    i = 1 
    while True: 
        new_id = f"#{i:05d}" 
        if new_id not in existing_ids:
             return new_id 
        i += 1

# LOGIN
def login_page():
    st.markdown('<div class="logo">Shopping</div>', unsafe_allow_html=True)
    st.markdown('<div class="title">ĐĂNG NHẬP</div>', unsafe_allow_html=True)

    user_id = st.text_input("Nhập User ID")
    st.markdown('<span style="color: #000000">Chưa có tài khoản?</span>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    USER_FILE = "App\\users.csv"

    if os.path.exists(USER_FILE):
        try:
            users_df = pd.read_csv(USER_FILE)
        except pd.errors.EmptyDataError:
            users_df = pd.DataFrame(columns=["user_id", "name", "gender", "age"])
    else:
        users_df = pd.DataFrame(columns=["user_id", "name", "gender", "age"])


    with col2:
        if st.button("Đăng nhập"):
            if not user_id.strip():
                st.warning("Vui lòng nhập User ID")
                return

            if user_id in df_log["user_id"].values:
                st.session_state.current_user = user_id
                st.session_state.user_info = {"user_id": user_id}
                st.session_state.page = "products"
                st.success("Đăng nhập thành công!")
                st.rerun()
            elif user_id in users_df["user_id"].values:
                st.session_state.current_user = user_id
                user_row = users_df[users_df["user_id"] == user_id].iloc[0]
                st.session_state.user_info = user_row.to_dict()
                st.session_state.page = "products"
                st.success("Đăng nhập thành công!")
                st.rerun()
            else:
                st.error("User ID không tồn tại")

    with col1:
        if st.button("Đăng ký"):
            st.session_state.page = "cold_start"
            st.rerun()

#COLD START)
def cold_start_page():
    st.markdown('<div class="logo">Shopping</div>', unsafe_allow_html=True)
    st.markdown('<div class="title">ĐĂNG KÝ</div>', unsafe_allow_html=True)

    name = st.text_input("Tên người dùng")
    birth_year = st.number_input(
        "Năm sinh", min_value=1900, max_value=datetime.now().year
    )
    gender = st.selectbox("Giới tính", ["Male", "Female"])

    if st.button("Đăng ký"):
        if name.strip() == "":
            st.warning("Vui lòng nhập tên")
            return

        new_user_id = generate_new_user_id(df_log["user_id"].unique())

        st.session_state.new_user_id = new_user_id
        st.session_state.user_info = {
            "user_id": new_user_id,
            "name": name,
            "gender": gender,
            "age": calculate_age(birth_year)
        }
        
        df_users = pd.DataFrame(columns=["user_id", "name", "gender", "age"])

        # Thêm user mới
        df_users = pd.concat([df_users, pd.DataFrame([st.session_state.user_info])], ignore_index=True)
        df_users.to_csv('App\\users.csv', index=False)

        st.success("🎉 Đăng ký thành công!")
        st.markdown(
            f"""
            <div class="user-box">
                🆔 <b>User ID của bạn:</b> <span style="color:#d32f2f">{new_user_id}</span><br>
                👤 {name} &nbsp;&nbsp;
                🎂 {calculate_age(birth_year)} tuổi &nbsp;&nbsp;
                🚻 {gender}
            </div>
            """,
            unsafe_allow_html=True
        )

    if st.session_state.new_user_id:
        if st.button("Đăng nhập"):
            st.session_state.page = "login"
            st.rerun()


# PAGE RECOMMEND
def product_page():
    st.markdown('<div class="logo">Shopping</div>', unsafe_allow_html=True)

    user_id = st.session_state.current_user
    
    col1, col2 = st.columns([3,1])

    with col1:
        st.markdown('<div class="tl_rec">Danh sách sản phẩm</div>', unsafe_allow_html=True)
    with col2:
        if st.button("Đăng xuất"):
            st.session_state.page = "login"
            st.rerun()
    user_logs = df_log[df_log["user_id"] == user_id]

    if user_logs.empty:

        if "user_info" in st.session_state:
            user_info = st.session_state.user_info
        else:
            user_info = df_users[df_users["user_id"] == user_id].iloc[0].to_dict()

        st.markdown(
        f"""
        <div class="user-box">
            👤 <b>{user_id}</b>
        </div>
        """,
        unsafe_allow_html=True
    )
        # Gọi hàm gợi ý cho user mới
        results = hybrid.recommend_cold_start(
            age=user_info["age"],
            gender=user_info["gender"],
            top_k=20
        )

        render_products(results)
        return

    user_row = (
        user_logs
        .sort_values("timestamp", ascending=False)
        .iloc[0]
    )

    st.markdown(
        f"""
        <div class="user-box">
            👤 <b>{user_id}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.spinner("Đang tạo gợi ý cá nhân hóa..."):
        results = hybrid.recommend(
            user_id=user_id,
            top_k=20
        )

    render_products(results)

if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "cold_start":
    cold_start_page()
elif st.session_state.page == "products":
    product_page()
