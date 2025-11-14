import pandas as pd
import joblib
from flask import Flask, request, render_template_string
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz

# ---------------------------------------------------------
# 1. Load dữ liệu gốc
# ---------------------------------------------------------
df = pd.read_csv("gdelt_cleaned_with_text.csv")
df = df.dropna(subset=["content"])
df = df.reset_index(drop=True)

# ---------------------------------------------------------
# 2. Load TF-IDF vectorizer (joblib)
# ---------------------------------------------------------
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ---------------------------------------------------------
# 3. Load ma trận TF-IDF (npz)
# ---------------------------------------------------------
tfidf_matrix = load_npz("tfidf_matrix.npz")

# ---------------------------------------------------------
# 4. Hàm gợi ý bài báo
# ---------------------------------------------------------
def recommend_news(input_text, top_n=5):
    # Vector hóa câu nhập vào
    input_vec = tfidf.transform([input_text])

    # Tính cosine similarity
    cosine_sim = cosine_similarity(input_vec, tfidf_matrix).flatten()

    # Lấy top các bài tương tự
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


# ---------------------------------------------------------
# 5. Flask App
# ---------------------------------------------------------
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Gợi ý tin tức</title>
</head>
<body>
    <h2>Hệ thống gợi ý tin tức theo nội dung</h2>

    <form action="/recommend" method="post">
        <textarea name="text" rows="6" cols="70"
        placeholder="Nhập nội dung tin tức để tìm bài tương tự..."></textarea><br><br>
        <button type="submit">Gợi ý</button>
    </form>

    {% if results %}
    <h2>Kết quả gợi ý</h2>
    {% for r in results %}
        <div style="margin-bottom:20px;">
            <b>Tiêu đề:</b> {{ r.title }} <br>
            <b>URL:</b> <a href="{{ r.url }}" target="_blank">{{ r.url }}</a> <br>
            <b>Độ tương đồng:</b> {{ r.similarity }} <br>
            <p>{{ r.content }}</p>
        </div>
        <hr>
    {% endfor %}
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/recommend", methods=["POST"])
def recommend():
    user_text = request.form["text"]
    results = recommend_news(user_text)
    return render_template_string(HTML_TEMPLATE, results=results)


if __name__ == "__main__":
    app.run(debug=True)
