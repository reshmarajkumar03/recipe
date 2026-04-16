import streamlit as st
from PIL import Image

def load_fixed_height(path, height=500):
    img = Image.open(path)
    ratio = height / img.height
    new_width = int(img.width * ratio)
    return img.resize((new_width, height), Image.LANCZOS)

st.title("Project Details")
st.subheader("Using Machine Learning to Classify Sentiment, Predict Ratings, and Recommend Recipes")
st.write(
    """
    This project uses review text and user interaction data to study recipe preferences.
    We apply three machine learning methods: a Text CNN for sentiment classification,
    a hybrid collaborative filtering model for recipe recommendation, and a Bayesian
    regression model for rating prediction.
    """
)
st.divider()

st.header("Methods Overview")

if "carousel_index" not in st.session_state:
    st.session_state.carousel_index = 0

slides = [
    {"image": "cnn.png", "title": "Text CNN", "desc": "Sentiment classification from review text"},
    {"image": "collab.png", "title": "Collaborative Filtering", "desc": "Hybrid similarity-based recommendations"},
    {"image": "bayes.png", "title": "Bayesian Regression", "desc": "Predicting user ratings"},
]

current = st.session_state.carousel_index
slide = slides[current]

# Title right under header
st.markdown(f"### {slide['title']}")
st.write(slide["desc"])

# Arrows on sides of image
left, mid, right = st.columns([1, 10, 1])
with left:
    st.write("")
    st.write("")
    if st.button("◀", use_container_width=True):
        st.session_state.carousel_index = (current - 1) % len(slides)
        st.rerun()
with mid:
    st.image(load_fixed_height(slide["image"], height=500), use_container_width=True)
with right:
    st.write("")
    st.write("")
    if st.button("▶", use_container_width=True):
        st.session_state.carousel_index = (current + 1) % len(slides)
        st.rerun()

# Small dots centered below
dots_html = "".join([
    f"<span style='font-size:10px; margin:0 3px; color:{'#555' if i == current else '#ccc'}'>●</span>"
    for i in range(len(slides))
])
st.markdown(
    f"<div style='text-align:center; padding-top:4px'>{dots_html}</div>",
    unsafe_allow_html=True
)

st.subheader("Method Summaries")
st.markdown(
    """
    **Text CNN**
    - Achieved approximately 80% accuracy in sentiment classification
    - Performed especially well on positive reviews, with strong precision and F1-scores
    - Captured useful sentiment patterns from review text for downstream recommendation tasks
    - Performance was lower for neutral and negative reviews because of class imbalance and ambiguity

    **Collaborative Filtering**
    - The initial item-item model performed poorly because user interaction data were sparse
    - The hybrid model produced higher and more meaningful similarity scores
    - Recommendations were more accurate and consistently grouped similar recipes
    - The model achieved a better balance between relevance and diversity

    **Bayesian Regression**
    - Achieved moderate predictive performance with RMSE = 1.27, MAE = 0.94, and R² = 0.36
    - Learned meaningful patterns in how users rate recipes
    - CNN-derived sentiment features improved predictions beyond engagement metrics alone
    - Demonstrated the value of combining structured interaction data with text-based sentiment features
    """
)
st.divider()
st.write("Use the Results page to explore the interactive recommendation system.")
