import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz

# ---------------------------------------
# Load dữ liệu
# ---------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("gdelt_cleaned_with_text.csv")
    df = df.dropna(subset=["content"])
    df = df.reset_index(drop=True)
    return df

@st.cache_resource
def load_model():
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    tfidf_matrix = load_npz("tfidf_matrix.npz")
    return tfidf, tfidf_matrix


df = load_data()
tfidf, tfidf_matrix = load_model()


# ---------------------------------------
# Hàm gợi ý
# ---------------------------------------
def recommend_news(input_text, top_n=5):
    input_vec = tfidf.transform([input_text])
    cosine_sim = cosine_similarity(input_vec, tfidf_matrix).flatten()
    top_idx = cosine_sim.argsort()[-top_n:][::-1]

    results = []
    for idx in top_idx:
        results.append({
            "title": df.loc[idx, "title"],
            "url": df.loc[idx, "SOURCEURL"],
            "content": df.loc[idx, "content"][:350] + "...",
            "similarity": round(float(cosine_sim[idx]), 4)
        })
    return results


# ---------------------------------------
# Giao diện Streamlit
# ---------------------------------------
st.title("🔍 Hệ thống gợi ý tin tức theo nội dung (TF-IDF Cosine Similarity)")

user_input = st.text_area("Nhập nội dung bài báo:", height=180)

if st.button("Gợi ý"):
    if user_input.strip() == "":
        st.warning("Vui lòng nhập nội dung!")
    else:
        results = recommend_news(user_input)

        st.subheader("🔎 Kết quả gợi ý")
        for r in results:
            st.markdown(f"""
            ### 📰 {r['title']}
            **URL:** [Link bài báo]({r['url']})  
            **Độ tương đồng:** `{r['similarity']}`  

            {r['content']}
            ---
            """)
