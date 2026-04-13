import streamlit as st
import pandas as pd

st.title("Recipe Recommendation System")

st.write("Select a category and a dish to get recommendations.")

# Load data
book3 = pd.read_excel("Book3.xlsx")
recipes = pd.read_csv("recipes.csv")

# Clean data
book3.columns = ["recipe_name", "category"]
book3["category"] = book3["category"].str.lower().str.strip()
book3["recipe_name"] = book3["recipe_name"].str.strip()

recipes["category"] = recipes["category"].str.lower().str.strip()
recipes["recipe_name"] = recipes["recipe_name"].str.strip()

# Dropdown 1: Category
categories = sorted(book3["category"].unique())
selected_category = st.selectbox("Pick a category", categories)

# Dropdown 2: Dish
filtered = book3[book3["category"] == selected_category]
dishes = sorted(filtered["recipe_name"].unique())

selected_dish = st.selectbox("Pick a dish from Book 3", dishes)

st.write("You selected:", selected_dish)
