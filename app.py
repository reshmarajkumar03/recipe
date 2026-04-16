import streamlit as st

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

col1, col2, col3 = st.columns(3)

st.header("Methods Overview")

col1, col2, col3 = st.columns(3)

# COLUMN 1 (keep same)
with col1:
    st.image("cnn.png")
    st.markdown("**Text CNN**")
    st.write("Sentiment classification from review text")

# COLUMN 2 (slightly bigger + align)
with col2:
    st.image("collab.png", width=420)
    st.write("")
    st.markdown("**Collaborative Filtering**")
    st.write("Hybrid similarity-based recommendations")

# COLUMN 3 (slightly bigger + align)
with col3:
    st.image("bayes.png", width=420)
    st.write("")
    st.markdown("**Bayesian Regression**")
    st.write("Predicting user ratings")

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
