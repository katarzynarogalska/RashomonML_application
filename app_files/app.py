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

DATASETS = {
    "HR Job Change" : "hr",
    "Credit Score" : "credit",
    "Breast Cancer" : "breast_cancer",
    "COMPAS" : "compas",
    "Heart Disease" : "heart",
    "Glass Types" : "glass",
    "Letter Recognition" : "letter_recognition",
    "Yeast" : "yeast"
}
css_path = current_dir/ "style.css"

st.set_page_config(
    page_title="App",
    page_icon="🏠",
    layout="wide"
)

with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


if 'global_dataset' not in st.session_state:
    st.session_state.global_dataset = "--choose--"

if 'weighted_sum_method' not in st.session_state:
    st.session_state.weighted_sum_method = None

if 'custom_weights' not in st.session_state:
    st.session_state.custom_weights = None
if 'metric1' not in st.session_state:
    st.session_state.metric1 = "--choose--"

if 'metric2' not in st.session_state:
    st.session_state.metric2 = "--choose--"

if 'task_type' not in st.session_state:
    st.session_state.task_type = None

if 'base_metric' not in st.session_state:
    st.session_state.base_metric = "--choose--"

if 'epsilon' not in st.session_state:
    st.session_state.epsilon = None

def get_page():
    return st.query_params.get("page", "home")

def set_page(page):
    st.query_params["page"] = page
    st.session_state.strona = page
    st.rerun()

if 'strona' not in st.session_state:
    st.session_state.strona = get_page()

def change_page(page):
    st.session_state.strona = page
    st.rerun()

def hide_sidebar():
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def top_nav():
    current = st.session_state.strona

    st.markdown(f"""
    <style>
        
        .top-tabs {{
            display: flex;
            gap: 2.5rem;
            padding: 0.8rem 2rem 0.4rem 2rem;
            border-bottom: 1px solid #e5e7eb;
            background-color: white;
            margin: 0;
            font-family: 'Inter', sans-serif;
            font-size: 1rem;
        }}
        .top-tabs .tab {{
            text-decoration: none;
            color: #6b7280;
            font-weight: 500;
            padding: 0.5rem 0;
        }}
        .top-tabs .tab:hover {{
            color: #16476A;
        }}
        .top-tabs .tab.active {{
            color: #16476A;
            border-bottom: 3px solid #16476A;
        }}
    </style>
    <div class="top-tabs">
        <a href="?page=home" target="_self" class="tab {'active' if current=='home' else ''}">Home</a>
        <a href="?page=datasets" target="_self" class="tab {'active' if current=='datasets' else ''}">Datasets</a>
        <a href="?page=rashomon" target="_self" class="tab {'active' if current=='rashomon' else ''}">Rashomon page</a>
        <a href="?page=intersection" target="_self" class="tab {'active' if current=='intersection' else ''}">Intersection page</a>
    </div>
    """, unsafe_allow_html=True)
# def datasets_sidebar():
#     options = ["--choose--"] + list(DATASETS.keys())

   
#     def update_dataset():
#         #st.session_state.global_dataset = st.session_state.temp_selectbox

#         if st.session_state.global_dataset in ["HR Job Change", "COMPAS", "Breast Cancer", "Heart Disease"]:
#             st.session_state.task_type = "binary"
#         elif st.session_state.global_dataset in ["Credit Score" , "Glass Types" , "Letter Recognition" ,"Yeast"]:
#             st.session_state.task_type ="multiclass"
#         else:
#             st.session_state.task_type = None
    
   
#     st.session_state.global_dataset = st.sidebar.selectbox(
#         "Choose a dataset to analyze",
#         options=options, 
#         key="temp_selectbox",  
#         index=options.index(st.session_state.global_dataset) if st.session_state.global_dataset in options else 0,
#         on_change=update_dataset
#         )
#     st.sidebar.markdown("---")

#     if st.session_state.global_dataset == "--choose--":
#         st.session_state.global_dataset = None

   
top_nav()
page = st.session_state.strona

# Home page configuration --------------------------------------------------------------------------------------
if page == "home":
    
    st.markdown('<div class="homepage_title"> Welcome to ARSA website </div>', unsafe_allow_html=True)
    st.markdown('<div class="homepage_subtitle"> Automated Rashomon Set Analysis </div>', unsafe_allow_html=True)
    
    
    col1,col2 = st.columns([0.5,2], vertical_alignment="center")
    with col1:  
            with st.container(key="logo"):
                    img=Image.open(root_path/"app_files"/"pics"/"logo.jpg")
                    st.image(img,width='stretch')
    with col2:
        
        st.markdown('<div class="section_title_overw"> 🖳 Application overview </div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section_descr_overw"> {front_page_descriptions.package_overview} </div>', unsafe_allow_html=True)

   
    # rashomon set definition
    st.markdown('<div class="section_title_rashomon_and_int"> &nbsp; The Rashomon Set concept </div>', unsafe_allow_html=True)
   
    col1, col2 = st.columns([1,1.5], vertical_alignment="center")
    with col2:
        st.markdown(f'<div class ="rashomon_style"> {front_page_descriptions.rashomon_set_definition} </div>', unsafe_allow_html=True)
    with col1:
        with st.container(key="left-margin"):
            img = Image.open(root_path/"app_files"/"pics"/"rashomon.jpg")
            st.image(img,width='stretch')
        
        
    st.markdown("""
        <style>
           /* 1. Zewnętrzny kontener – marginesy */
[data-testid="stExpander"] {
    margin-top: 1rem !important;
    margin-left: 2.5rem !important;
    margin-right: 2.5rem !important;
   
                
}

/* 2. Wnętrze ekspandera – tło, border i padding */
[data-testid="stExpander"] > details {
    border: 1px solid #16476A !important;
    border-radius: 0.6rem !important;
    padding: 1rem !important;             /* wypełnienie */
    background-color: #16476A !important;
               
}

/* 3. Label ekspandera */
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {
    color: white!important;
    font-size: 1.3vw !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    padding:0.01vw;
}

/* 4. Ikona ekspandera */
[data-testid="stExpander"] summary svg {
    width: 28px !important;
    height: 28px !important;
    color: white !important;
    fill: white !important;
}

    </style>
    """, unsafe_allow_html=True)
    with st.expander(label="Why is that so important?", expanded=False):
        st.markdown(f'<div class ="rashomon_style_question_white"> {front_page_descriptions.rashomon_set_situation} </div>', unsafe_allow_html=True)

    st.markdown('<div class="section_title_rashomon_and_int "> &nbsp; 🖩 The Rashomon Set metrics </div>', unsafe_allow_html=True)
    st.markdown('<div class="section_descr_metrics"> You can expand this section to access an intuitive description of metrics that were used to assess the differences between models and the Rashomon Set properties. Formal definitions can be found in the articles linked at the bottom of this page. </div>', unsafe_allow_html=True)

    st.markdown("""
        <style>
           /* 1. Zewnętrzny kontener – marginesy */
[data-testid="stExpander"] {
    margin-top: 0rem !important;
    margin-left: 2.5rem !important;
    margin-right: 2.5rem !important;
   
                
}

/* 2. Wnętrze ekspandera – tło, border i padding */
[data-testid="stExpander"] > details {
    border: 1px solid #16476A !important;
    border-radius: 0.6rem !important;
    padding: 1rem !important;             /* wypełnienie */
    background-color: #16476A !important;
               
}

/* 3. Label ekspandera */
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {
    color: white!important;
    font-size: 1.3vw !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    padding:0.01vw;
}

/* 4. Ikona ekspandera */
[data-testid="stExpander"] summary svg {
    width: 28px !important;
    height: 28px !important;
    color: white !important;
    fill: white !important;
}

    </style>
    """, unsafe_allow_html=True)
    with st.expander(label="Expand for details →", expanded=False):
        st.markdown(f'<div class="section_descr"> {front_page_descriptions.rashomon_metrics} </div>', unsafe_allow_html=True)

    st.markdown("---")

    # Intersection ovewview ------------------------------------------------
    st.markdown('<div class="section_title_rashomon_and_int"> &nbsp; The Rashomon Intersection concept </div>', unsafe_allow_html=True)
   
    col1, col2 = st.columns([1,1.5], vertical_alignment="center")
    with col2:
        st.markdown(f'<div class ="rashomon_style"> {front_page_descriptions.rashomon_intersection_definition} </div>', unsafe_allow_html=True)
    with col1:
        with st.container(key="left-margin_2"):
            img = Image.open(root_path/"app_files"/"pics"/"intersection.jpg")
            st.image(img,width='stretch')

    st.markdown('<div class="section_title_rashomon_and_int "> &nbsp; 🖩 The Rashomon Intersection metrics </div>', unsafe_allow_html=True)
    st.markdown('<div class="section_descr_metrics">As the Rashomon Intersection is defined as the intersection of two Rashomon Sets, all metrics from the Rashomon Set also apply. In addition, we compare the Rashomon Intersection with traditional methods for analyzing the multi-objective optimization problems, such as the Pareto Front.  </div>', unsafe_allow_html=True)

    #bibliography
    st.markdown('<div class="section_title_rashomon_and_int "> &nbsp; 🕮 References </div>', unsafe_allow_html=True)
    st.markdown('<div class="section_descr_metrics"> This section provides links to articles and books that were used while creating the package and analyzing the Rashomon Set properties. <br> Please expand for more details. </div>', unsafe_allow_html=True)

    st.markdown("""
        <style>
           /* 1. Zewnętrzny kontener – marginesy */
[data-testid="stExpander"] {
    margin-top: 0rem !important;
    margin-left: 2.5rem !important;
    margin-right: 2.5rem !important;
   
                
}

/* 2. Wnętrze ekspandera – tło, border i padding */
[data-testid="stExpander"] > details {
    border: 1px solid #16476A !important;
    border-radius: 0.6rem !important;
    padding: 1rem !important;             /* wypełnienie */
    background-color: #16476A !important;
               
}

/* 3. Label ekspandera */
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {
    color: white!important;
    font-size: 1.3vw !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    padding:0.01vw;
}

/* 4. Ikona ekspandera */
[data-testid="stExpander"] summary svg {
    width: 28px !important;
    height: 28px !important;
    color: white !important;
    fill: white !important;
}

    </style>
    """, unsafe_allow_html=True)
    with st.expander(label="Expand to see bibliography →", expanded=False):
        st.markdown(f'<div class="section_descr"> {front_page_descriptions.bibliography} </div>', unsafe_allow_html=True)
  

            

    

# Datasets page configuration ------------------------------------------------------------------------------------
elif page == "datasets":
    hide_sidebar()
    datasets_page.show()

elif page == "rashomon":
    
   
    rashomon_page.show()


elif page == "intersection":
    
       
    intersection_page.show()
    

