import streamlit as st

st.title("Recipe Recommendation System")

st.write("An interactive application that recommends recipes using a hybrid similarity model.")

# -----------------------------
# About
# -----------------------------
st.header("About This Project")

st.write("""
Online recipe platforms provide access to a large number of dishes, making it easier to explore new meals but also more difficult to decide what to cook. 
This project addresses that problem by building a recommendation system that suggests similar recipes based on content, category, and user engagement.

The goal is to generate meaningful and relevant recommendations even when user interaction data is limited.
""")

# -----------------------------
# Data
# -----------------------------
st.header("Data")

st.write("""
The dataset used is the Recipe Reviews and User Feedback Dataset from the UCI Machine Learning Repository. 
It contains recipe information, user reviews, ratings, and engagement metrics such as likes and reply counts.

Additional preprocessing steps were performed, including:
- Cleaning recipe text and removing noise
- Converting relevant columns to numeric format
- Creating a category column to improve structure and avoid incorrect groupings
""")

# -----------------------------
# Method
# -----------------------------
st.header("Method")

st.write("""
A hybrid recommendation approach was used instead of relying on a single method. 
Three types of similarity were combined:

- **Text similarity (70%)**: Computed using TF-IDF to capture similarities in ingredients and descriptions
- **Category similarity (20%)**: Provides a boost for recipes in the same or related categories
- **Engagement similarity (10%)**: Based on user interactions such as likes and ratings

These components were combined into a final similarity matrix (`hybrid_sim`), allowing the model to balance content relevance with contextual and behavioral signals.
""")

# -----------------------------
# Results Preview
# -----------------------------
st.header("Results Preview")

st.write("""
The similarity score distribution demonstrates how each component behaves. 
Text similarity produces mostly low values, showing strong differentiation between recipes, while engagement similarity is concentrated at higher values and is less discriminative. 
The hybrid similarity shows a more balanced spread, indicating a good mix of the components.

This confirms that the weighting allows text to drive similarity, while category and engagement provide supporting influence, resulting in more meaningful recommendations.
""")

# -----------------------------
# Charts
# -----------------------------
st.subheader("Visualizations")

st.write("Similarity Score Distribution:")
st.image("similarity_chart.png")  # replace with your actual file name

st.write("Second Chart:")
st.image("second_chart.png")  # replace with your second chart

# -----------------------------
# Navigation note
# -----------------------------
st.info("Go to the Results page from the sidebar to explore recommendations.")
