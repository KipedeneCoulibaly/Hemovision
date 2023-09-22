import pandas as pd
import plotly.express as px
import streamlit as st


title        = "Hemovision"
sidebar_name = "Exploratory Data Analysis"


def run():
    st.title(title)
    st.divider()
    st.write("# 📊 Exploratory Data Analysis")
    st.divider()

    df = pd.read_csv("streamlit_app/data/df/PBC_dataset_norm.csv")
    st.markdown("### Database")
    st.dataframe(df[["filepath","label"]].head())

    st.markdown("### Descriptive Statistics")
    st.dataframe(df.describe(include=["O"]))

    st.markdown("### Distributions")
    fig_distrib = px.histogram(df, x='label',template='none',color='label')
    st.container().plotly_chart(fig_distrib)