import streamlit as st
import pandas as pd
import numpy as np
import re
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

st.title("Recipe Recommendation System")
st.write("Choose a category and a dish to see the top 3 recommendations.")

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

# -----------------------------
# Build metadata
# -----------------------------
recipe_meta = recipes[["recipe_code", "recipe_name", "category"]].drop_duplicates()

GENERIC_WORDS = [
    "recipe", "make", "made", "use", "used", "great", "good", "delicious",
    "easy", "really", "just", "like", "love", "loved", "time", "way",
    "little", "added", "add", "also", "well", "try", "tried", "family",
    "everyone", "taste", "tasted", "tastes", "nice", "got", "did",
    "came", "turned", "definitely", "highly", "recommend", "perfect",
    "wonderful", "amazing", "excellent", "better", "best", "instead",
    "sure", "want", "bit", "lot", "pretty", "think", "thought",
    "minutes", "minute", "hour", "hours", "cup", "cups", "tablespoon",
    "teaspoon", "oven", "pan", "bowl", "pot", "dish",
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

@st.cache_data
def build_recommender(recipes_df):
    recipe_text = (
        recipes_df.groupby("recipe_code")["text"]
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

    engagement = recipes_df.groupby("recipe_code").agg(
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

    common_recipes = text_sim_df.index.intersection(engagement_sim_df.index)
    categories_series = (
        recipe_meta.set_index("recipe_code")["category"]
        .reindex(common_recipes)
        .fillna("other")
    )

    n = len(common_recipes)
    cat_array = categories_series.values
    cat_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if cat_array[i] == cat_array[j]:
                cat_matrix[i, j] = 1.0
            elif cat_array[j] in RELATED_CATEGORIES.get(cat_array[i], []):
                cat_matrix[i, j] = 0.5

    cat_sim_df = pd.DataFrame(cat_matrix, index=common_recipes, columns=common_recipes)

    text_aligned = text_sim_df.loc[common_recipes, common_recipes]
    eng_aligned = engagement_sim_df.loc[common_recipes, common_recipes]

    hybrid_sim = (
        0.70 * text_aligned +
        0.20 * cat_sim_df +
        0.10 * eng_aligned
    )

    return hybrid_sim

hybrid_sim = build_recommender(recipes)

def recommend(recipe_code, n=3):
    if recipe_code not in hybrid_sim.index:
        return pd.DataFrame()

    scores = hybrid_sim[recipe_code].drop(index=recipe_code)
    top_n = scores.nlargest(n).reset_index()
    top_n.columns = ["recipe_code", "similarity_score"]
    top_n = top_n.merge(recipe_meta, on="recipe_code", how="left")

    seed_cat = recipe_meta.loc[recipe_meta["recipe_code"] == recipe_code, "category"].values[0]

    def tag(row):
        if row["category"] == seed_cat:
            return "same"
        elif row["category"] in RELATED_CATEGORIES.get(seed_cat, []):
            return "related"
        return "cross"

    top_n["match_type"] = top_n.apply(tag, axis=1)
    return top_n[["recipe_name", "category", "match_type", "similarity_score"]]

# -----------------------------
# Dropdown 1: category from recipes.csv
# -----------------------------
category_options = ["Select a category"] + sorted(recipe_meta["category"].dropna().unique().tolist())
selected_category = st.selectbox("Pick a category", category_options, index=0)

# -----------------------------
# Dropdown 2: dish from recipes.csv
# -----------------------------
selected_dish = None

if selected_category != "Select a category":
    filtered_dishes = (
        recipe_meta[recipe_meta["category"] == selected_category]["recipe_name"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    dish_options = ["Select a dish"] + filtered_dishes
    selected_dish = st.selectbox("Pick a dish", dish_options, index=0)
else:
    st.selectbox("Pick a dish", ["Select a category first"], index=0, disabled=True)

# -----------------------------
# Show nothing until both are selected
# -----------------------------
if selected_category == "Select a category" or selected_dish in [None, "Select a dish"]:
    st.stop()

# -----------------------------
# Match selected dish to recipe_code
# -----------------------------
matches = recipe_meta[
    (recipe_meta["recipe_name"].str.lower() == selected_dish.lower().strip()) &
    (recipe_meta["category"] == selected_category)
].copy()

if matches.empty:
    st.warning("This dish was not found in the main recipes dataset.")
    st.stop()

selected_code = matches.iloc[0]["recipe_code"]
recs = recommend(selected_code, n=3)

if recs.empty:
    st.warning("No recommendations found.")
    st.stop()

st.subheader(f"Top 3 recommendations for {selected_dish}")
st.write(
    "These recommendations are based on a hybrid model that combines text similarity, "
    "category similarity, and engagement."
)

st.dataframe(recs, use_container_width=True)

# -----------------------------
# Fixed-axis chart: 0 to 1
# -----------------------------
chart = (
    alt.Chart(recs)
    .mark_bar()
    .encode(
        x=alt.X("recipe_name:N", sort="-y", title="Recommended Recipe"),
        y=alt.Y("similarity_score:Q", scale=alt.Scale(domain=[0, 1]), title="Similarity Score"),
        tooltip=["recipe_name", "category", "match_type", "similarity_score"]
    )
    .properties(height=350)
)

st.altair_chart(chart, use_container_width=True)
