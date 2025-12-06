import streamlit as st
from pathlib import Path
import os
import sys
root_path = Path(__file__).parent.parent.parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from description_files.datasets_descr import return_datasets_descriptions


def display_data_info(data_dict, idx):
    
    with st.container(key=f"white_cont{idx}"):
        st.markdown(f'<div class="dataset_title"> &nbsp; {data_dict["Title"]}</div>', unsafe_allow_html=True)
        
        for key, value in data_dict.items():

            if key=="Title":
                continue
            if key=="Feature description":
                continue
            else:

                st.markdown(f'<div class="key_style">{key}</div>', unsafe_allow_html=True)
                if key == "Characteristics":
                    st.markdown(f'<div class="data_descr add_bottom_padding">{data_dict[key]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="data_descr">{data_dict[key]}</div>', unsafe_allow_html=True)



def show():
    st.markdown('<div class="homepage_title"> Datasets Page </div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section_descr"> This page is dedicated to provide detailed descriptions and characteristics of pre-saved datasets that are available for analysis. Additionally, we present charts illustrating the characteristics mentioned in the descriptions. </div>', unsafe_allow_html=True)
    list_dict = return_datasets_descriptions()
    for i,dict in enumerate(list_dict):
        display_data_info(dict, i)
