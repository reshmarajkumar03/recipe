import streamlit as st
import pandas as pd

st.title("Results")

st.write("This page will contain an interactive visualization.")

data = pd.DataFrame({
    "Method": ["A", "B", "C"],
    "Score": [0.5, 0.6, 0.7]
})

selected_method = st.selectbox("Choose a method", data["Method"])

st.write("You selected:", selected_method)

st.write("Sample Results Table:")
st.dataframe(data)

st.write("Sample Chart:")
st.bar_chart(data.set_index("Method"))
