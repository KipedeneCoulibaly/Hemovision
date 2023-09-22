from PIL import Image
from streamlit_app.utils import ui
import numpy as np
from streamlit_app.utils.models import models as my_models
import streamlit as st


title        = "Hemovision"
sidebar_name = "Predictions"


def run():
    st.title(title)
    st.divider()
    st.write("# 📈 Prediction")
    st.divider()
    model_list = ["ResNet34_8c", "ResNet34_11c"]
    st.header('Prediction')

    # I)
    st.write('')
    st.subheader('1. Choose which model you want to use for prediction')
    model_choice = st.selectbox("Select a model :", options=model_list,)
    st.write(model_choice)

    # II)
    st.write('')
    st.subheader('2. Upload an image or select a preloaded example')
    st.markdown('*Note: please remove any uploaded image to choose an example image from the list.*')
    
    cola, colb, colc = st.columns([4, 1, 4])
    with cola:
        uploaded_img = ui.upload_example()
    with colb:
        st.markdown('''<h3 style='text-align: center;'><br><br>OR</h3>''', unsafe_allow_html=True)
    with colc:
        dict_img, example_path = ui.select_examples()

    if uploaded_img:
        selected_img = "streamlit_app/data/images/uploaded/" + uploaded_img.name
        img_name =  uploaded_img.name
        img_file = open(selected_img, 'rb')
    elif example_path:
        selected_img = dict_img[example_path]
        img_file = open(selected_img, 'rb')
        try:
            img_name = img_file.name.replace("\\", "/").split("/")[-1]
        except IndexError:
            st.write('please load an image.')

    img = img_file.read()
    
    img_info = Image.open(img_file)
    file_details = f"""
                    Name: {img_name}
                    Type: {img_info.format}
                    Size: {img_info.size}"""  
    
    # III)
    st.write('')
    st.subheader('3. Results')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original image ...")
        st.image(img, width=150)
        st.caption(file_details)
        st.write("")
    with col2:
        with st.container():
            st.subheader("... is probably :")
            if model_choice == "ResNet34_8c":
                model_class8,classname8,_,_,images_crop = my_models.Hemovision(img_file.name)
                y_pred08      = model_class8.predict(images_crop)
                y_pred8       = np.argmax(y_pred08, axis=-1)
                class8        = classname8[int(y_pred8[0])]
                st.write(class8)

            if model_choice == "ResNet34_11c":
                _,_,model_class11, classname11, images_crop = my_models.Hemovision(img_file.name)
                y_pred011     = model_class11.predict(images_crop)
                y_pred11      = np.argmax(y_pred011, axis=-1)
                class11       = classname11[int(y_pred11[0])]
                st.write(class11)