import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

st.title("Recipe Recommendation Demo")

st.write(
    "Choose a category and a dish from Book 3 to see the top 3 recipe recommendations."
)

# Load data
@st.cache_data
def load_data():
    recipes = pd.read_csv("recipes.csv")
    book3 = pd.read_excel("Book3.xlsx")

    # Clean Book3 column names
    book3.columns = ["recipe_name", "category"]
    book3["recipe_name"] = book3["recipe_name"].astype(str).str.strip()
    book3["category"] = book3["category"].astype(str).str.strip().str.lower()

    return recipes, book3

recipes, book3 = load_data()

# Build recommender
@st.cache_data
def build_recommender(recipes):
    recipes = recipes.copy()

    for col in ["thumbs_up", "best_score", "reply_count", "stars"]:
        recipes[col] = pd.to_numeric(recipes[col], errors="coerce").fillna(0)

    recipes["text"] = recipes["text"].fillna("")
    recipes["recipe_name"] = recipes["recipe_name"].astype(str).str.strip()
    recipes["category"] = recipes["category"].astype(str).str.strip().str.lower()

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

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    recipe_text = (
        recipes.groupby("recipe_code")["text"]
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
        stop_words=stop_words,
    )
    tfidf_matrix = tfidf.fit_transform(recipe_text["combined_text"])

    text_sim = cosine_similarity(tfidf_matrix)
    text_sim_df = pd.DataFrame(
        text_sim,
        index=recipe_text["recipe_code"],
        columns=recipe_text["recipe_code"],
    )

    engagement = recipes.groupby("recipe_code").agg(
        avg_thumbs_up=("thumbs_up", "mean"),
        avg_best_score=("best_score", "mean"),
        avg_reply_count=("reply_count", "mean"),
        avg_stars=("stars", "mean"),
        total_reviews=("user_id", "count"),
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
        columns=engagement["recipe_code"],
    )

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

    common_recipes = text_sim_df.index.intersection(engagement_sim_df.index)
    categories = (
        recipe_meta.set_index("recipe_code")["category"]
        .reindex(common_recipes)
        .fillna("other")
    )

    n = len(common_recipes)
    cat_array = categories.values
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

    return recipe_meta, hybrid_sim, RELATED_CATEGORIES

recipe_meta, hybrid_sim, RELATED_CATEGORIES = build_recommender(recipes)

# Recommendation function
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

# Dropdown 1: category
categories = sorted(book3["category"].dropna().unique())
selected_category = st.selectbox("Pick a category", categories)

# Dropdown 2: dish from Book 3
filtered_book3 = book3[book3["category"] == selected_category].copy()
dish_options = sorted(filtered_book3["recipe_name"].dropna().unique())

selected_dish = st.selectbox("Pick a dish from Book 3", dish_options)

# Map dish name to recipe_code
matches = recipe_meta[
    (recipe_meta["recipe_name"].str.lower() == selected_dish.lower().strip()) &
    (recipe_meta["category"] == selected_category)
].copy()

if matches.empty:
    st.warning("That Book 3 dish was not found in the main recipes data.")
else:
    selected_code = matches.iloc[0]["recipe_code"]

    st.subheader(f"Top 3 recommendations for: {selected_dish}")

    recs = recommend(selected_code, n=3)

    if recs.empty:
        st.warning("No recommendations found.")
    else:
        st.dataframe(recs, use_container_width=True)

        st.write("Recommendation scores")
        chart_df = recs.set_index("recipe_name")[["similarity_score"]]
        st.bar_chart(chart_df)
