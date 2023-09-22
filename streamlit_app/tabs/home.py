
from PIL import Image
import streamlit as st


title        = "Hemovision"
subtitle     = "Automated blood cells classification for diagnosis and research"
sidebar_name = "Home"

def run():
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")

    st.title(title)
    st.subheader(subtitle, divider="red")
    st.write("# 🏠 Home")

    st.divider()
    st.markdown("""
                ### Context
                Blood is made up of 3 main cell types.  These cells are generally subjected to morphological analysis by blood 
                experts in order to detect possible blood diseases. This is a risky, time-consuming task if these practitioners 
                don't have the right tools.
                """)
    st.markdown("""
                ### Aim
                The goal of this project is to to identify the different types of blood cells using computer vision algorithms. 
                The density and relative abundance of blood cells in the smear is crucial for the diagnosis of many pathologies, 
                such as leukemia, which relies on the ratio of lymphocytes. The identification of abnormal leukocytes in 
                pathologies such as leukemia could complete this first part. Developing a tool capable of analyzing cells from 
                blood smears could facilitate the diagnosis of certain pathologies, but could also be used for research purposes.
                """)
    
    st.image(Image.open("streamlit_app/data/images/ResNet34/ia.png"), use_column_width=True)
    st.image(Image.open("streamlit_app/data/images/blood_nn.png"),use_column_width=True)
    st.markdown("""
                ### Data
                The database used for our studies comes from the central laboratory of the Barcelona clinical hospital, and 
                represents a set of 17,092 digital images of cells obtained from blood smears.   
                The [PBC_dataset_normal_Barcelona dataset](https://data.mendeley.com/datasets/snkd93bnjr/draft?a=d9582c71-9af0-4e59-9062-df30df05a121) is
                publicly accessible and contains the following cells:
                """)
    st.image(Image.open("streamlit_app/data/images/blood_cells.png"), use_column_width=True, 
             caption="Images of different types of leukocytes: neutrophil, eosinophil, basophil, lymphocyte, monocyte and\
                    immature granulocytes (metamyelocyte, myelocyte and promyelocyte), erythroblasts and platelets (thrombocytes)\
                    that can be found in peripheral blood and some of their most important morphological characteristics.")
    
    
    