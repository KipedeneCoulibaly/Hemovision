from collections import OrderedDict
import streamlit as st
import streamlit_app.config as config
from streamlit_app.tabs import home, EDA, modelisation, prediction, about

st.set_page_config(
    page_title=config.TITLE,
    page_icon="streamlit_app/data/images/logo-hemovision.png",
    layout="wide",
)

with open("streamlit_app/style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

TABS = OrderedDict(
    [
        (home.sidebar_name, home),
        (EDA.sidebar_name, EDA),
        (modelisation.sidebar_name, modelisation),
        (prediction.sidebar_name, prediction),
        (about.sidebar_name, about),
    ]
)

def run():
    st.sidebar.image("streamlit_app/data/images/logo-hemovision.png",width=200,)
    tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.success(
        """
        This app is maintained by the Hemovision team. You can learn more about me at
        [Hemovison📧](https://github.com/juin23_bcds_blood_cells/).
        """
    )

    st.sidebar.markdown("### Team members:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]

    tab.run()


if __name__ == "__main__":
    run()
