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
        [data-testid="stExpander"] details summary p,
        [data-testid="stExpander"] details summary div,
        .st-emotion-cache-1cpxqw2 {
            font-size: 1.2vw !important;
            color: #557d95!important;
            font-weight: 500 !important;
            font-family: 'Inter', sans-serif !important;
        }

        /* 2. USUŃ BORDER Z CAŁEGO EXPANDERA */
        [data-testid="stExpander"] {
            border: none !important;
            background-color: transparent !important;
            box-shadow: none !important;
            margin-left: 3rem !important;
            margin-right:3rem !important;
            padding: 0 !important;
        }

        /* 3. USUŃ BORDER Z WNĘTRZA */
        [data-testid="stExpander"] > div,
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] details > div {
            border: none !important;
            background-color: transparent !important;
            box-shadow: none !important;
        }
                [data-testid="stExpander"] .st-emotion-cache-1hynsf2,
    [data-testid="stExpander"] .st-emotion-cache-1p1m4ay,
    [data-testid="stExpander"] details > div:last-child,
    [data-testid="stExpander"] .streamlit-expanderContent {
        background-color: #16476A !important; /* Jasnoniebieski */
        border-radius: 0.6rem !important;
        padding: 1.5rem !important;
        margin-top: 0.5rem !important;
        margin-bottom: 1rem !important;
    }
[data-testid="stExpander"] .stMarkdown,
    [data-testid="stExpander"] .stPlotlyChart,
    [data-testid="stExpander"] .stColumns {
        background-color: transparent !important;
    }

    /* 5. STYL DLA POSZCZEGÓLNYCH ELEMENTÓW */
    [data-testid="stExpander"] .key_style,
    [data-testid="stExpander"] .data_descr,
    [data-testid="stExpander"] .dataset_title {
        background-color: transparent !important;
        color: white !important;
    }

    /* 6. IKONA EXPANDERA */
    [data-testid="stExpander"] summary svg {
        width: 28px !important;
        height: 28px !important;
        color: #16476A !important;
        fill: #16476A!important;
    }

    /* 7. USUŃ DODATKOWE TŁA Z ELEMENTÓW STREAMLIT */
    [data-testid="stExpander"] .st-emotion-cache-1v0mbdj,
    [data-testid="stExpander"] .st-emotion-cache-1v0mbdj > div {
        background-color: transparent !important;
    }

    /* 8. DLA WSZYSTKICH ELEMENTÓW W EXPANDERZE */
    [data-testid="stExpander"] * {
        background-color: transparent !important;
        color: white !important;
    }

    /* 9. SPECJALNY STYL DLA CAŁEJ ZAWARTOŚCI */
    .expander-content-wrapper {
        background-color: #e6f0ff !important;
        padding: 20px !important;
        border-radius: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)
        
    with st.expander("Expand for details →", expanded=False):    
        for key, value in data_dict.items():

            if key=="Title" or key== "path" or key =="target_col":
                continue
            if key=="Feature description":
                continue
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
