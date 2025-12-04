import streamlit as st
import sys
from pathlib import Path
from other_pages import rashomon_page, intersection_page, datasets_page
root_path = Path(__file__).parent.parent
texts_path = Path(__file__).parent.parent / "description_files" 
sys.path.append(str(texts_path))
import front_page_descriptions
from PIL import Image
import pandas as pd

current_dir = Path(__file__).parent

MULTICLASS_METRICS = ["accuracy", "balanced_accuracy", "precision_macro", "precision_micro", "precision_weighted",
    "recall_macro", "recall_micro", "recall_weighted", "f1_macro", "f1_micro", "f1_weighted", "roc_auc_ovo",
    "roc_auc_ovo_weighted", "roc_auc_ovr", "roc_auc_ovr_micro", "roc_auc_ovr_weighted"]

BINARY_METRICS = ["accuracy", "balanced_accuracy", "roc_auc", "average_precision", "precision", "precision_macro",
    "precision_micro", "precision_weighted", "recall", "recall_macro", "recall_micro", "recall_weighted",
    "f1", "f1_macro", "f1_micro", "f1_weighted"]

css_path = current_dir/ "style.css"

st.set_page_config(
    page_title="App",
    page_icon="🏠",
    layout="wide"
)

with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if 'strona' not in st.session_state:
    st.session_state.strona = 'home'

def change_page(page):
    st.session_state.strona = page
    st.rerun()

def global_nav():
    st.sidebar.markdown('<div class="sidebar_title">Menu</div>', unsafe_allow_html=True)

    if st.sidebar.button("Home", use_container_width=True):
        change_page("home")

    if st.sidebar.button("Datasets"):
        change_page("datasets")

    if st.sidebar.button("Rashomon page", use_container_width=True):
        change_page("rashomon")

    if st.sidebar.button("Intersection page", use_container_width=True):
        change_page("intersection")

    st.sidebar.markdown("---")

# Home page configuration --------------------------------------------------------------------------------------
if st.session_state.strona == "home":
    global_nav()
    st.markdown('<div class="homepage_title"> Welcome to RashoML </div>', unsafe_allow_html=True)
    st.markdown('<div class="homepage_subtitle"> Compare. Understand. Decide. </div>', unsafe_allow_html=True)
    st.markdown(f'<div class="overview_descr"> {front_page_descriptions.front_page_overview} </div>', unsafe_allow_html=True)
    st.markdown('<div class="section_title"> 🖳 Application overview </div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section_descr"> {front_page_descriptions.package_overview} </div>', unsafe_allow_html=True)
    st.markdown('<div class="section_title"> The Rashomon Set concept </div>', unsafe_allow_html=True)
    with st.container(key="white_cont"):
        col1, col2 = st.columns(2)
        with col2:
            st.markdown(f'<div class ="rashomon_style"> {front_page_descriptions.rashomon_set_definition} </div>', unsafe_allow_html=True)
        with col1:

            img = Image.open(root_path/"app_files"/"pics"/"rashomon.jpg")
            st.image(img)

        st.markdown(f'<div class ="rashomon_question"> Why is that so important? </div>', unsafe_allow_html=True)
        st.markdown(f'<div class ="rashomon_style_question"> {front_page_descriptions.rashomon_set_situation} </div>', unsafe_allow_html=True)

    st.markdown('<div class="section_title no_margin"> 🕮 The Rashomon Set metrics </div>', unsafe_allow_html=True)
    st.markdown('<div class="section_descr_metrics"> You can expand this section to access an intuitive description of metrics that were used to assess the differences between models and the Rashomon Set properties. Formal definitions can be found in the articles linked at the bottom of this page. </div>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    [data-testid="stExpander"] details summary p,
    [data-testid="stExpander"] details summary div,
    .st-emotion-cache-1cpxqw2 {
        font-size: 1.4vw !important;
        color: #16476A !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* 2. USUŃ BORDER Z CAŁEGO EXPANDERA */
    [data-testid="stExpander"] {
        border: none !important;
        background-color: transparent !important;
        box-shadow: none !important;
        margin: 0 !important;
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

    /* 4. STYL DLA ZAWARTOSCI EXPANDERA */
    [data-testid="stExpander"] .section_descr {
        background-color: white;
        border-radius: 0.6rem;
        padding: 1.2vw;
        margin-top: 0vw;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }

    /* 5. IKONA EXPANDERA */
    [data-testid="stExpander"] summary svg {
        width: 28px !important;
        height: 28px !important;
        color: #16476A !important;
        fill: #16476A !important;
    }
    </style>
""", unsafe_allow_html=True)
    with st.expander(label="Expand for details →", expanded=False):
        
        st.markdown(f'<div class="section_descr"> {front_page_descriptions.rashomon_metrics} </div>', unsafe_allow_html=True)


# Datasets page configuration ------------------------------------------------------------------------------------
elif st.session_state.strona == "datasets":
    global_nav()
    datasets_page.show()

elif st.session_state.strona == "rashomon":
    global_nav()
    #najpierw uzytkownik niech wybierze zbior danych i na podstawie tego bedzie mial opcje base metric z opcji binary albo multiclass

    task_type = 'binary'
    if task_type == 'binary':
        base_metric = st.sidebar.selectbox("Choose base metric:", BINARY_METRICS)
    elif task_type == 'multiclass':
        base_metric =  st.sidebar.selectbox("Choose base metric:", MULTICLASS_METRICS)

    epsilon = st.sidebar.slider("Choose epsilon:", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

    rashomon_page.show()


elif st.session_state.strona == "intersection":
    global_nav()

    task_type = 'binary'
    if task_type == 'binary':
        metric1 = st.sidebar.selectbox("Choose first metric:", BINARY_METRICS)
        possible_metric2 = [m for m in BINARY_METRICS if m != metric1]
        metric2 = st.sidebar.selectbox("Choose second metric:", possible_metric2)
    elif task_type == 'multiclass':
        metric1 = st.sidebar.selectbox("Choose first metric:", MULTICLASS_METRICS)
        possible_metric2 = [m for m in MULTICLASS_METRICS if m != metric1]
        metric2 = st.sidebar.selectbox("Choose second metric:", possible_metric2)

    metrics = [metric1, metric2]

    epsilon = st.sidebar.slider("Choose epsilon:", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

    weighted_sum_method = st.sidebar.radio("Choose weighted sum method:", ["custom_weights", "entropy", "critic"])
    if weighted_sum_method == 'custom_weights':
        weight1 = weight2 = None
        #czy tu jest opcja zeby sie zmienialy automatycznie jedna i druga tak zeby sie sumowaly do 1?
        st.sidebar.write("Set custom weights:")
        weight1 = st.sidebar.number_input(f"Weight for metric {metric1}:", min_value=0.0, value=0.5, step=0.01)
        weight2 = st.sidebar.number_input(f"Weight for metric {metric2}:", min_value=0.0, value=0.5, step=0.01)
        custom_weights = (weight1, weight2)

    intersection_page.show()
    

