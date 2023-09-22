import streamlit as st


title        = "Hemovision"
sidebar_name = "Modelisation"


def run():
    st.title(title)
    st.divider()
    st.write("# 💻 Modelisation")
    st.divider()
    st.write("### Cells classification")
    st.image("streamlit_app/data/images/ResNet34/cells_classification_context.jpg")
    st.write("### Pipeline")
    st.image("streamlit_app/data/images/ResNet34/pipeline.png")
    st.write("### Segmentation model : U-net")
    st.image("streamlit_app/data/images/ResNet34/u_net.png")
    st.write("### Classification model : ResNet34")
    st.image("streamlit_app/data/images/ResNet34/resnet34.jpg")
    st.write("")
