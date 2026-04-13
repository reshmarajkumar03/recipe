import streamlit as st
import pandas as pd
import numpy as np
import re
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

st.title("🍽️ Recipe Recommendation System")
st.write(
    "Select a category and a dish to generate the top 3 recipe recommendations."
)

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    recipes = pd.read_csv("recipes.csv")

    recipes["category"] = recipes["category"].astype(str).str.lower().str.strip()
    recipes["recipe_name"] = recipes["recipe_name"].astype(str).str.strip()
    recipes["text"] = recipes["text"].fillna("")

    for col in ["thumbs_up", "best_score", "reply_count", "stars"]:
        recipes[col] = pd.to_numeric(recipes[col], errors="coerce").fillna(0)

    return recipes

recipes = load_data()
recipe_meta = recipes[["recipe_code", "recipe_name", "category"]].drop_duplicates()

# -----------------------------
# Helper settings
# -----------------------------
GENERIC_WORDS = [
    "recipe", "make", "made", "use", "used", "great", "good", "delicious",
    "easy", "really", "just", "like", "love", "loved", "time", "way",
    "little", "added", "add", "also", "well", "try", "tried", "family",
    "everyone", "taste", "tasted", "tastes", "nice", "got", "did",
    "came", "turned", "definitely", "highly", "recommend", "perfect",
    "wonderful", "amazing", "excellent", "better", "best", "instead",
    "sure", "want", "bit", "lot", "pretty", "think", "thought",
    "minutes", "minute", "hour", "hours", "cup", "cups", "tablespoon",
    "teaspoon", "oven", "pan", "bowl", "pot", "dish"
]

RELATED_CATEGORIES = {
    "dessert": ["bread", "breakfast"],
    "bread": ["dessert", "breakfast"],
    "breakfast": ["bread", "dessert"],
    "soup": ["casserole", "chicken", "beef"],
    "chicken": ["soup", "casserole", "pasta"],
    "pasta": ["chicken", "casserole", "beef"],
    "beef": ["pasta", "soup", "casserole"],
    "casserole": ["beef", "chicken", "pasta", "soup"],
    "seafood": ["soup", "pasta"],
    "salad": ["side", "chicken"],
    "side": ["salad", "casserole"],
    "pork": ["beef", "casserole"],
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Build recommender
# -----------------------------
@st.cache_data
def build_recommender(df, recipe_meta_df):
    recipe_text = (
        df.groupby("recipe_code")["text"]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )
    recipe_text.columns = ["recipe_code", "combined_text"]
    recipe_text["combined_text"] = recipe_text["combined_text"].apply(clean_text)

    stop_words = list(TfidfVectorizer(stop_words="english").get_stop_words()) + GENERIC_WORDS

    tfidf = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=2,
        stop_words=stop_words
    )
    tfidf_matrix = tfidf.fit_transform(recipe_text["combined_text"])

    text_sim = cosine_similarity(tfidf_matrix)
    text_sim_df = pd.DataFrame(
        text_sim,
        index=recipe_text["recipe_code"],
        columns=recipe_text["recipe_code"]
    )

    engagement = df.groupby("recipe_code").agg(
        avg_thumbs_up=("thumbs_up", "mean"),
        avg_best_score=("best_score", "mean"),
        avg_reply_count=("reply_count", "mean"),
        avg_stars=("stars", "mean"),
        total_reviews=("user_id", "count")
    ).reset_index()

    for col in ["avg_thumbs_up", "avg_best_score", "avg_reply_count", "total_reviews"]:
        engagement[col] = np.log1p(engagement[col])

    scaler = MinMaxScaler()
    engagement_scaled = scaler.fit_transform(
        engagement[["avg_thumbs_up", "avg_best_score", "avg_reply_count", "avg_stars", "total_reviews"]]
    )

    engagement_sim = cosine_similarity(engagement_scaled)
    engagement_sim_df = pd.DataFrame(
        engagement_sim,
        index=engagement["recipe_code"],
        columns=engagement["recipe_code"]
    )

    common = text_sim_df.index.intersection(engagement_sim_df.index)

    cat_series = (
        recipe_meta_df.set_index("recipe_code")["category"]
        .reindex(common)
        .fillna("other")
    )

    n = len(common)
    cat_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if cat_series.iloc[i] == cat_series.iloc[j]:
                cat_matrix[i, j] = 1.0
            elif cat_series.iloc[j] in RELATED_CATEGORIES.get(cat_series.iloc[i], []):
                cat_matrix[i, j] = 0.5

    cat_sim_df = pd.DataFrame(cat_matrix, index=common, columns=common)

    hybrid = (
        0.70 * text_sim_df.loc[common, common] +
        0.20 * cat_sim_df +
        0.10 * engagement_sim_df.loc[common, common]
    )

    return hybrid

hybrid_sim = build_recommender(recipes, recipe_meta)

# -----------------------------
# Recommendation function
# -----------------------------
def recommend(recipe_code, n=3):
    if recipe_code not in hybrid_sim.index:
        return pd.DataFrame()

    scores = hybrid_sim[recipe_code].drop(index=recipe_code)
    top = scores.nlargest(n).reset_index()
    top.columns = ["recipe_code", "similarity_score"]
    top = top.merge(recipe_meta, on="recipe_code", how="left")

    seed_cat = recipe_meta.loc[
        recipe_meta["recipe_code"] == recipe_code, "category"
    ].values[0]

    def label_match(row):
        if row["category"] == seed_cat:
            return "Same category"
        elif row["category"] in RELATED_CATEGORIES.get(seed_cat, []):
            return "Related category"
        return "Different category"

    top["match_type"] = top.apply(label_match, axis=1)

    return top[["recipe_name", "category", "match_type", "similarity_score"]]

# -----------------------------
# User inputs
# -----------------------------
category_options = ["Select a category"] + sorted(
    recipe_meta["category"].dropna().unique().tolist()
)
selected_category = st.selectbox("Pick a category", category_options, index=0)

selected_dish = None

if selected_category != "Select a category":
    dishes = (
        recipe_meta[recipe_meta["category"] == selected_category]["recipe_name"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    dish_options = ["Select a dish"] + dishes
    selected_dish = st.selectbox("Pick a dish", dish_options, index=0)
else:
    st.selectbox("Pick a dish", ["Select a category first"], index=0, disabled=True)

# -----------------------------
# Keep page blank until selection
# -----------------------------
if selected_category == "Select a category" or selected_dish in [None, "Select a dish"]:
    st.stop()

# -----------------------------
# Match selected dish to recipe_code
# -----------------------------
match = recipe_meta[
    (recipe_meta["recipe_name"].str.lower() == selected_dish.lower().strip()) &
    (recipe_meta["category"] == selected_category)
]

if match.empty:
    st.warning("Dish not found in the dataset.")
    st.stop()

selected_code = match.iloc[0]["recipe_code"]
recs = recommend(selected_code, n=3)

if recs.empty:
    st.warning("No recommendations found.")
    st.stop()

# -----------------------------
# Results section
# -----------------------------
st.divider()
st.subheader(f"Top 3 recommendations for {selected_dish}")

st.markdown(
    "These recommendations come from a hybrid model that combines **text similarity**, "
    "**category similarity**, and **engagement features**."
)

# Make scores look cleaner
display_recs = recs.copy()
display_recs["similarity_score"] = display_recs["similarity_score"].round(3)

# Summary cards
for i, row in display_recs.reset_index(drop=True).iterrows():
    st.markdown(
        f"""
**#{i+1} {row['recipe_name']}**  
Category: {row['category'].title()}  
Match type: {row['match_type']}  
Similarity score: {row['similarity_score']}
"""
    )

st.write("")
st.dataframe(display_recs, use_container_width=True)

# -----------------------------
# Fixed-axis chart
# -----------------------------
chart = (
    alt.Chart(display_recs)
    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
    .encode(
        x=alt.X("recipe_name:N", sort="-y", title="Recommended Recipe"),
        y=alt.Y(
            "similarity_score:Q",
            scale=alt.Scale(domain=[0, 1]),
            title="Similarity Score"
        ),
        tooltip=[
            alt.Tooltip("recipe_name:N", title="Recipe"),
            alt.Tooltip("category:N", title="Category"),
            alt.Tooltip("match_type:N", title="Match Type"),
            alt.Tooltip("similarity_score:Q", title="Score")
        ]
    )
    .properties(height=350)
)

st.altair_chart(chart, use_container_width=True)
