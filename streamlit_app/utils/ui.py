from pathlib import Path
import streamlit as st
import glob
import os

path_img   = Path('streamlit_app/data/images/app_dataset/')

fname_dict = {'SNE': 'segmented neutrophil',
              'EO': 'eosinophil',
              'BA': 'basophil',
              'LY': 'lymphocyte',
              'MO': 'monocyte',
              'PLT': 'platelet',
              'ERB': 'erythroblast',
              'MMY': 'metamyelocyte',
              'MY': 'myelocyte',
              'PMY': 'promyelocyte',
              'BNE': 'band neutrophil'}

def uploaded_dir_manager(limit):
    if limit < 0: raise ValueError("limit must be a positif number")
    """"
    Deletes the contents of the downloaded file folder when it reaches a size limit
    parameters:
        limit: number of images.
    """
    path = "streamlit_app/data/images/uploaded/"
    if len(os.listdir(path)) == limit:
        files = glob.glob(path+"*")
        for f in files: os.remove(f)

def save_uploaded_file(uploadedfile):
    """
    Saves images in a temporary folder
    parameters:
        uploadedfile: file to upload 
    """
    with open(os.path.join("streamlit_app/data/images/uploaded",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())

def select_examples():
    """Image selection form"""
    folders_img = [f'{i.stem} - {fname_dict[i.stem]}' for i in path_img.iterdir()]
    class_img   = st.selectbox('a. Select a cell type', options=folders_img)
    class_path  = path_img / class_img.split(' -')[0]

    dict_img     = {image_path.stem: image_path for image_path in class_path.iterdir()}
    example_path = st.selectbox('b. Select an example image', options=dict_img.keys())
    return dict_img, example_path

def upload_example():
    """external image download"""
    img_file = st.file_uploader("Upload an image for classification", type=['jpg', 'png'], )
    if img_file:
        uploaded_dir_manager(10)
        save_uploaded_file(img_file)
    return img_file

