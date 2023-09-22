import streamlit as st


title        = "Hemovision"
sidebar_name = "About"


def run():
    st.title(title)
    st.divider()
    st.write("# 🤔 About")
    st.divider()
    st.write("### Contributors")
    st.markdown("""
    Bertrand-Elie DURAN

    Sergey SASNOUSKI

    Joseph LIEBER
                
    Kipédène Coulibaly
                """)

    st.markdown("""

    ### References

    ##### Data :
                
    *[peripheral blood cell images](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7182702/)*
                
    *[Acute Promyelocytic Leukemia](https://www.kaggle.com/eugeneshenderov/acute-promyelocytic-leukemia-apl)*  
   
    ##### Bibliography:
                
    *[Andrea Acevedo and al (2019)](https://www.sciencedirect.com/science/article/abs/pii/S0169260719303578?via%3Dihub)*
                
    *[Laura Boldú and al (2001)](https://www.sciencedirect.com/science/article/abs/pii/S0169260721000742?via%3Dihub)*

    """, unsafe_allow_html=True)
    st.write(""),st.write("")

