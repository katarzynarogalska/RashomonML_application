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


if 'strona' not in st.session_state:
    st.session_state.strona = 'home'

if 'global_dataset' not in st.session_state:
    st.session_state.global_dataset = None

if 'weighted_sum_method' not in st.session_state:
    st.session_state.weighted_sum_method = None

if 'custom_weights' not in st.session_state:
    st.session_state.custom_weights = None
if 'metric1' not in st.session_state:
    st.session_state.metric1 = None

if 'metric2' not in st.session_state:
    st.session_state.metric2 = None

if 'task_type' not in st.session_state:
    st.session_state.task_type = None

if 'base_metric' not in st.session_state:
    st.session_state.base_metric = None

if 'epsilon' not in st.session_state:
    st.session_state.epsilon = None

def change_page(page):
    st.session_state.strona = page
    st.rerun()

def global_nav():
    st.sidebar.markdown('<div class="sidebar_title">Menu</div>', unsafe_allow_html=True)

    if st.sidebar.button("Home", use_container_width=True):
        change_page("home")

    if st.sidebar.button("Datasets", use_container_width=True):
        change_page("datasets")

    if st.sidebar.button("Rashomon page", use_container_width=True):
        change_page("rashomon")

    if st.sidebar.button("Intersection page", use_container_width=True):
        change_page("intersection")

    st.sidebar.markdown("---")
   
    options = ["--choose--"] + list(DATASETS.keys())

    def update_dataset():
        #st.session_state.global_dataset = st.session_state.temp_selectbox

        if st.session_state.global_dataset in ["HR Job Change", "COMPAS", "Breast Cancer", "Heart Disease"]:
            st.session_state.task_type = "binary"
        elif st.session_state.global_dataset in ["Credit Score" , "Glass Types" , "Letter Recognition" ,"Yeast"]:
            st.session_state.task_type ="multiclass"
        else:
            st.session_state.task_type = None
    
   
    st.session_state.global_dataset = st.sidebar.selectbox(
        "Choose a dataset to analyze",
        options=options, 
        key="temp_selectbox",  
        index=options.index(st.session_state.global_dataset) if st.session_state.global_dataset in options else 0,
        on_change=update_dataset
        )
    st.sidebar.markdown("---")

    if st.session_state.global_dataset == "--choose--":
        st.session_state.global_dataset = None

   

# Home page configuration --------------------------------------------------------------------------------------
if st.session_state.strona == "home":
    global_nav()
    st.markdown('<div class="homepage_title"> Welcome to ARSA  </div>', unsafe_allow_html=True)
    st.markdown('<div class="homepage_subtitle"> Automated Rashomon Set Analysis  </div>', unsafe_allow_html=True)
    st.markdown(f'<div class="overview_descr"> {front_page_descriptions.front_page_overview} </div>', unsafe_allow_html=True)
    st.markdown('<div class="section_title"> 🖳 Application overview </div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section_descr"> {front_page_descriptions.package_overview} </div>', unsafe_allow_html=True)

    with st.container(key = "rashomon_background"):
        st.markdown('<div class="section_title_rashomon_and_int"> &nbsp; The Rashomon Set concept </div>', unsafe_allow_html=True)
        with st.container(key="white_cont"):
            col1, col2 = st.columns(2)
            with col2:
                st.markdown(f'<div class ="rashomon_style"> {front_page_descriptions.rashomon_set_definition} </div>', unsafe_allow_html=True)
            with col1:

                img = Image.open(root_path/"app_files"/"pics"/"rashomon.jpg")
                st.image(img)
                
          
            with st.container(key = "blue_cont"):
                    with st.expander(label="Why is that so important?", expanded=False):
                        st.markdown(f'<div class ="rashomon_style_question_white"> {front_page_descriptions.rashomon_set_situation} </div>', unsafe_allow_html=True)

        st.markdown('<div class="section_title_rashomon_and_int no_margin"> &nbsp; 🖩 The Rashomon Set metrics </div>', unsafe_allow_html=True)
        st.markdown('<div class="section_descr_metrics"> You can expand this section to access an intuitive description of metrics that were used to assess the differences between models and the Rashomon Set properties. Formal definitions can be found in the articles linked at the bottom of this page. </div>', unsafe_allow_html=True)

        st.markdown("""
        <style>
        [data-testid="stExpander"] details summary p,
        [data-testid="stExpander"] details summary div,
        .st-emotion-cache-1cpxqw2 {
            font-size: 1.4vw !important;
            color: white !important;
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
            color: white !important;
            fill: white !important;
        }
        </style>
    """, unsafe_allow_html=True)
        with st.expander(label="Expand for details →", expanded=False):
            
            st.markdown(f'<div class="section_descr"> {front_page_descriptions.rashomon_metrics} </div>', unsafe_allow_html=True)

        

    # Intersection ovewview ------------------------------------------------
    with st.container(key = "intersection_background"):
        st.markdown('<div class="section_title_rashomon_and_int"> &nbsp; The Rashomon Intersection concept </div>', unsafe_allow_html=True)
        with st.container(key="white_cont_int"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<div class ="intersection_style"> {front_page_descriptions.rashomon_intersection_definition} </div>', unsafe_allow_html=True)
            with col2:

                img = Image.open(root_path/"app_files"/"pics"/"intersection.jpg")
                st.image(img)
        st.markdown('<div class="section_title_rashomon_and_int no_margin"> &nbsp; 🖩 The Rashomon Intersection metrics </div>', unsafe_allow_html=True)
        st.markdown('<div class="section_descr_metrics">As the Rashomon Intersection is defined as the intersection of two Rashomon Sets, all metrics from the Rashomon Set also apply. In addition, we compare the Rashomon Intersection with traditional methods for analyzing the multi-objective optimization problems, such as the Pareto Front.  </div>', unsafe_allow_html=True)

    with st.container(key = "bibliography_background"):
        st.markdown('<div class="section_title_rashomon_and_int"> &nbsp; 🕮 References </div>', unsafe_allow_html=True)
        st.markdown('<div class="section_descr_metrics"> This section provides links to articles and books that were used while creating the package and analyze the Rashomon Set properties. Please expand for more details. </div>', unsafe_allow_html=True)
        st.markdown("""
        <style>
        [data-testid="stExpander"] details summary p,
        [data-testid="stExpander"] details summary div,
        .st-emotion-cache-1cpxqw2 {
            font-size: 1.4vw !important;
            color: white !important;
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
            color: white !important;
            fill: white !important;
        }
        </style>
    """, unsafe_allow_html=True)
        with st.expander(label="Expand for bibliography →", expanded=False):
            st.markdown(f'<div class="section_descr"> {front_page_descriptions.bibliography} </div>', unsafe_allow_html=True)
            

       

# Datasets page configuration ------------------------------------------------------------------------------------
elif st.session_state.strona == "datasets":
    global_nav()
    datasets_page.show()

elif st.session_state.strona == "rashomon":
    global_nav()
    #najpierw uzytkownik niech wybierze zbior danych i na podstawie tego bedzie mial opcje base metric z opcji binary albo multiclass

    if st.session_state.global_dataset is not None:
       
        st.markdown("""
        <style>
        [data-testid="stExpander"] details summary p,
        [data-testid="stExpander"] details summary div,
        .st-emotion-cache-1cpxqw2 {
            font-size: 1.3vw !important;
            color: #16476A !important;
            font-weight: 500 !important;
            font-family: 'Inter', sans-serif !important;
            width: 90% !important;                
            margin: 0.2rem 0 !important;          
        
            cursor: pointer;
            }
        [data-testid="stExpander"] {
            margin: 0 !important;
            padding: 0 !important;
        }

        /* 2. USUŃ BORDER Z CAŁEGO EXPANDERA */
        [data-testid="stExpander"] {
            border: solid 1px lightgray !important;
            border-radius : 6px;
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
            background-color: blue;
            border-radius: 0.6rem;
            padding: 0vw;
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
        with st.sidebar.expander("Specify parameters", expanded=False):
            if st.session_state.global_dataset in ["HR Job Change", "COMPAS", "Breast Cancer", "Heart Disease"]:
                st.session_state.task_type = "binary"
            elif st.session_state.global_dataset in ["Credit Score", "Glass Types", "Letter Recognition", "Yeast"]:
                st.session_state.task_type = "multiclass"
            else:
                st.session_state.task_type = None

            # Teraz ustaw opcje dla base_metric
            if st.session_state.task_type is not None:
                metrics_options = ["-- choose --"] + (BINARY_METRICS if st.session_state.task_type == 'binary' else MULTICLASS_METRICS)
                
                st.session_state.base_metric = st.selectbox(
                    "Base metric:",
                    options=metrics_options,
                    index=0,  # domyślnie "-- choose --"
                    key="base_metric_select"
                )

                if st.session_state.base_metric == "-- choose --":
                    st.session_state.base_metric = None
            else:
                st.sidebar.write("Please choose a dataset first.")

            st.session_state.epsilon = st.slider("Epsilon:", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
   
    rashomon_page.show()


elif st.session_state.strona == "intersection":
    global_nav()

    if st.session_state.global_dataset is not None:
       
        st.markdown("""
        <style>
        [data-testid="stExpander"] details summary p,
        [data-testid="stExpander"] details summary div,
        .st-emotion-cache-1cpxqw2 {
            font-size: 1.3vw !important;
            color: #16476A !important;
            font-weight: 500 !important;
            font-family: 'Inter', sans-serif !important;
            width: 90% !important;                
            margin: 0.2rem 0 !important;          
        
            cursor: pointer;
            }
        [data-testid="stExpander"] {
            margin: 0 !important;
            padding: 0 !important;
        }

        /* 2. USUŃ BORDER Z CAŁEGO EXPANDERA */
        [data-testid="stExpander"] {
            border: solid 1px lightgray !important;
            border-radius : 6px;
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
            background-color: blue;
            border-radius: 0.6rem;
            padding: 0vw;
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
        with st.sidebar.expander("Specify parameters", expanded=False):
            if st.session_state.global_dataset in ["HR Job Change", "COMPAS", "Breast Cancer", "Heart Disease"]:
                st.session_state.task_type = "binary"
            elif st.session_state.global_dataset in ["Credit Score", "Glass Types", "Letter Recognition", "Yeast"]:
                st.session_state.task_type = "multiclass"
            else:
                st.session_state.task_type = None

            metrics_options = ["-- choose --"] + (BINARY_METRICS if st.session_state.task_type == 'binary' else MULTICLASS_METRICS)
                
            m1 = st.selectbox("First metric:", metrics_options, index=0, key="metric1_select")
            st.session_state.metric1 = None if m1 == "-- choose --" else m1

            possible_metric2 = ["-- choose --"] + [m for m in metrics_options[1:] if m != st.session_state.metric1]
            metric2_value = st.selectbox("Second metric:", possible_metric2, index=0, key="metric2_select")
            st.session_state.metric2 = None if metric2_value == "-- choose --" else metric2_value
           

            st.session_state.epsilon = st.slider("Epsilon:", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

            st.session_state.weighted_sum_method = st.radio("Weighted sum method:", ["custom_weights", "entropy", "critic"])

            if st.session_state.weighted_sum_method == "custom_weights":
                st.write("Set custom weights:")
                
                weight1 = st.number_input(f"Weight for {st.session_state.metric1 or 'Metric 1'}:", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="weight1_input")
                # Druga waga automatycznie uzupełnia do 1
                weight2 = 1.0 - weight1
                st.number_input(f"Weight for {st.session_state.metric2 or 'Metric 2'}:", min_value=0.0, max_value=1.0, value=weight2, step=0.01, key="weight2_input", disabled=True)
                st.session_state.custom_weights = (weight1, weight2)

    intersection_page.show()
    

