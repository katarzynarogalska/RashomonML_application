import streamlit as st
from pathlib import Path
from other_pages import rashomon_page, intersection_page
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
    st.sidebar.markdown("### Nawigacja")

    if st.sidebar.button("Home", use_container_width=True):
        change_page("home")

    if st.sidebar.button("Rashomon page", use_container_width=True):
        change_page("rashomon")

    if st.sidebar.button("Intersection page", use_container_width=True):
        change_page("intersection")

    st.sidebar.markdown("---")

if st.session_state.strona == "home":
    global_nav()
    st.title("Main page")
    st.write("""
        Tutaj overwiew i potem osobne sekcje na rashomon i intersection. 
        """)

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
    

'''

# Sidebar - nawigacja
st.sidebar.title("Menu")

if st.sidebar.button("Home", use_container_width=True, key="home"):
    st.session_state.strona = 'home'

if st.sidebar.button("Rashomon page", use_container_width=True, key="rashomon"):
    st.session_state.strona = 'rashomon'

if st.sidebar.button("Intersection page", use_container_width=True, key="intersection"):
    st.session_state.strona = 'intersection'

st.sidebar.markdown("---")

# Wyświetlanie stron
if st.session_state.strona == 'home':
    st.title("Main page")
    st.write("""
    Tutaj overwiew i potem osobne sekcje na rashomon i intersection. 
    
    """)
    
elif st.session_state.strona == 'rashomon':
    rashomon_page.show()
    
elif st.session_state.strona == 'intersection':
    intersection_page.show()
'''