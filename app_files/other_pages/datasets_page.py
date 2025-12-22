import streamlit as st
from pathlib import Path
import os
import sys
import pandas as pd
root_path = Path(__file__).parent.parent.parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from description_files.datasets_descr import return_datasets_descriptions
from description_files.dataset_plots import plot_class_distribution, plot_correlation_matrix, return_missing


def display_data_info(data_dict, idx):
    
    
    st.markdown(f'<div class="dataset_title"> &nbsp; {data_dict["Title"]}</div>', unsafe_allow_html=True)

    st.markdown("""
        <style>
        /* 1. Zewnętrzny kontener – marginesy */
        [data-testid="stExpander"] {
            margin-top: 0rem !important;
            margin-left: 2.5rem !important;
            margin-right: 2.5rem !important;
            max-width: calc(100% - 5rem) !important;  /* uwzględnia marginesy */
        }

        /* 2. Wnętrze ekspandera – tło, border i padding */
        [data-testid="stExpander"] > details {
            border: 1px solid #16476A !important;
            border-radius: 0.6rem !important;
            padding: 1rem !important;             
            background-color: #16476A !important;      
        }

        /* 3. Label ekspandera – tylko główny tekst, ignoruje kbd i inne dzieci */
        [data-testid="stExpander"] summary >span  {
            color: white !important;
            font-size: 1.3vw !important;
            font-weight: 500 !important;
            font-family: 'Inter', sans-serif !important;
            padding: 0 !important;
        }
        [data-testid="stExpander"] summary kbd {
            display: none !important;
        }

        /* 4. Ikona ekspandera */
        [data-testid="stExpander"] summary svg {
            width: 28px !important;
            height: 28px !important;
            color: white !important;
            fill: white !important;
            vertical-align: middle;
        }
        </style>
        """, unsafe_allow_html=True)
        
    with st.expander("Expand for details →", expanded=False):    
        for key, value in data_dict.items():

            if key=="Title" or key== "path" or key =="target_col":
                continue
            if key=="Feature description":
                st.markdown(f'<div class="key_style">{key}</div>', unsafe_allow_html=True) 
                list_html = "<ul>" 
                for feat in value: 
                    list_html += f"<li>{feat}</li>" 
                list_html += "</ul>" 
               
                st.markdown(
                    f'<div style="color: white;">  <style scoped>  div ul li {{  color: white !important;  margin-bottom: 0.5em; }}  </style>  {list_html}  </div>', 
                      unsafe_allow_html=True )
                continue

            if key=="Source":
                st.markdown(f'<div class="key_style">{key}</div>', unsafe_allow_html=True)
                st.markdown(
                f'''
                <div class="source-container-{idx}">
                    {data_dict[key]}
                </div>
                <style scoped>
                    .source-container-{idx} a {{
                        color: white !important;   /* kolor tylko dla tego linku */
                        text-decoration: none;
                        margin-left:0.5rem;
                    }}
                    .source-container-{idx} a:hover {{
                        color: white !important;
                        text-decoration: underline;
                    }}
                </style>
                ''', 
                unsafe_allow_html=True
            )
            

            else:

                st.markdown(f'<div class="key_style">{key}</div>', unsafe_allow_html=True)
                if key == "Characteristics":
                    st.markdown(f'<div class="data_descr add_bottom_padding">{data_dict[key]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="data_descr">{data_dict[key]}</div>', unsafe_allow_html=True)
        dataset = pd.read_csv(root_path/"datasets"/data_dict["path"])
        target = data_dict['target_col']
        nan_percent = return_missing(dataset)
        st.markdown(f'<div class="data_descr"> Percent of missing values : {nan_percent:.2f}%</div>', unsafe_allow_html=True)
        fig = plot_class_distribution(dataset, target)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig)
        with col2:
            corr_m = plot_correlation_matrix(dataset)
            st.plotly_chart(corr_m)
       



def show():
    st.markdown('<div class="homepage_title"> Datasets Page </div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section_descr"> This page is dedicated to provide detailed descriptions and characteristics of pre-saved datasets that are available for analysis. Additionally, we present charts illustrating the characteristics mentioned in the descriptions. </div>', unsafe_allow_html=True)
    list_dict = return_datasets_descriptions()
    for i,dict in enumerate(list_dict):
        display_data_info(dict, i)
