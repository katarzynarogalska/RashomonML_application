import streamlit as st
from rashomon_analysis.rashomon_set import RashomonSet
from rashomon_analysis.visualizers.rashomon_visualizer import Visualizer
from .dashboard.rashomon_binary import render_binary_dashboard
from .dashboard.rashomon_multiclass import render_multiclass_dashboard
import inspect 
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import random

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

MULTICLASS_METRICS = ["accuracy", "balanced_accuracy", "precision_macro", "precision_micro", "precision_weighted",
    "recall_macro", "recall_micro", "recall_weighted", "f1_macro", "f1_micro", "f1_weighted", "roc_auc_ovo",
    "roc_auc_ovo_weighted", "roc_auc_ovr", "roc_auc_ovr_micro", "roc_auc_ovr_weighted"]

BINARY_METRICS = ["accuracy", "balanced_accuracy", "roc_auc", "average_precision", "precision", "precision_macro",
    "precision_micro", "precision_weighted", "recall", "recall_macro", "recall_micro", "recall_weighted",
    "f1", "f1_macro", "f1_micro", "f1_weighted"]

if "task_type" not in st.session_state:
    st.session_state.task_type = None

def load_converted_data(selected_dataset, framework):
    dataset = DATASETS[selected_dataset]
    converter_results_path = Path(f"converter_results/{framework}/{dataset}")
    leaderboard = pd.read_csv( converter_results_path / "leaderboard.csv")
    y_true = pd.read_csv(converter_results_path/ "y_true.csv")
    with open(converter_results_path/ "predictions_dict.pkl", "rb") as f:
        predictions_dict = pickle.load(f)
    with open(converter_results_path / "proba_predictions_dict.pkl", "rb") as f:
        proba_predictions_dict = pickle.load(f)

    if (converter_results_path/"feature_importance_dict.pkl").is_file():
        with open(converter_results_path / "feature_importance_dict.pkl", "rb") as f:
            feature_importance_dict = pickle.load(f)
    else:
        feature_importance_dict = None
    return leaderboard, y_true, predictions_dict, proba_predictions_dict, feature_importance_dict


def show():

        st.markdown('<div class="homepage_title"> &nbsp; The Rashomon Set Analysis </div>', unsafe_allow_html=True)
        st.markdown("")
        with st.container(key="params"):
            st.markdown('<div class="params_title"> &nbsp; Specify the necessary parameters : </div>', unsafe_allow_html=True)
  
            col0,col1,col11, col2, col22, col3, col4, col5, col6 = st.columns([0.2,1,0.5,1,0.5,1,0.2,1,0.2])
            with col1:
                options = ["--choose--"] + list(DATASETS.keys())
                selected_dataset = st.selectbox(
                    "Dataset",
                    options,
                    index=0
                )
                if selected_dataset!="--choose--":
                    st.markdown(f'<div class="selected_params"> Selected dataset : {selected_dataset} </div>', unsafe_allow_html=True)
            with col2:
                if selected_dataset!="--choose--":
                    if selected_dataset in ["HR Job Change", "COMPAS", "Breast Cancer", "Heart Disease"]:
                        st.session_state.task_type = "binary"
                    elif selected_dataset in ["Credit Score", "Glass Types", "Letter Recognition", "Yeast"]:
                        st.session_state.task_type="multiclass"
                    metrics_options = ["--choose--"] + (BINARY_METRICS if st.session_state.task_type == 'binary' else MULTICLASS_METRICS)
                    selected_metric= st.selectbox(
                    "Base metric:",
                    options=metrics_options,
                    index=0,  
                    key="base_metric_select"
                    )
                    if selected_metric!="--choose--":
                        st.markdown(f'<div class="selected_params"> Selected metric : {selected_metric} </div>', unsafe_allow_html=True)

            with col3:
                if selected_dataset!="--choose--":
                    epsilon = st.slider("Epsilon:", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
                    st.markdown(f'<div class="selected_params"> Selected epsilon : {epsilon} </div>', unsafe_allow_html=True)
            with col5:
                if st.session_state.task_type =="binary":
                    delta = st.slider("delta:", 0.0, 1.0, 0.5, 0.01)
                    st.markdown(f'<div class="selected_params"> Selected delta : {delta} </div>', unsafe_allow_html=True)
        
        all_params_set = (selected_dataset != "--choose--" and selected_metric != "--choose--")

    
        st.markdown("""
        <style>
        /* 1. Zewnętrzny kontener – marginesy */
        [data-testid="stExpander"] {
            margin-top: 0rem !important;
            margin-left: 0rem !important;
            margin-right: 0rem !important;
            padding:0.1vw !important;
        }

        /* 2. Wnętrze ekspandera – tło, border i padding */
        [data-testid="stExpander"] > details {
            border : none !important;
            padding: 0.1rem !important;             
            background-color: rgb(0,0,0,0)!important;      
        }

        /* 3. Label ekspandera – tylko główny tekst, ignoruje kbd i inne dzieci */
        [data-testid="stExpander"] summary >span  {
            color: #16476A !important;
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
        with st.expander("Explore epsilon parameter"):
            if all_params_set:
                col1, col2 = st.columns([2,3])
                with col1:
                    framework_choice = st.radio("Select framework to analyze", ["AutoGluon", "H2O"], horizontal = False, key="vol")
                    framework_lower = framework_choice.lower()
                leaderboard, y_true, predictions_dict, proba_predictions_dict, feature_importance_dict = load_converted_data(selected_dataset, framework_lower)
                fig = visualize_rashomon_set_volume(leaderboard, predictions_dict, proba_predictions_dict, selected_metric)
                with col2:
                    st.pyplot(fig)
            else:
                st.markdown("Select a dataset and the base metric first.")

            




        

# autogluon tab -----------------------------------------------------------------------------------
        autogluon_tab, h2o_tab = st.tabs(["AutoGluon", "H2O"])

        with autogluon_tab:
            if all_params_set:
                leaderboard_autogluon, y_true_autogluon, predictions_dict_autogluon, proba_predictions_dict_autogluon, feature_importance_dict_autogluon = load_converted_data(selected_dataset, "autogluon")
                spinner_placeholder = st.empty()
                spinner_placeholder.markdown("""
                <div style="
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 200px;
                ">
                    <div style="
                        border: 4px solid #f3f3f3; 
                        border-top: 4px solid #426c85; 
                        border-radius: 50%;
                        width: 40px;
                        height: 40px;
                        animation: spin 1s linear infinite;
                    "></div>
                </div>

                <style>
                @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
                }
                </style>
                """, unsafe_allow_html=True)

                try:
                    rs_autogluon = RashomonSet(leaderboard = leaderboard_autogluon, predictions =predictions_dict_autogluon, proba_predictions = proba_predictions_dict_autogluon, feature_importances = feature_importance_dict_autogluon, base_metric = selected_metric, epsilon = epsilon)
                    visualizer_autogluon = Visualizer(rs_autogluon, y_true_autogluon)
                    plots_autogluon={}
                    descriptions_autogluon={}

                    if rs_autogluon.task_type == "binary":
                        #autoglon plots
                        st.session_state.task_type = "binary"
                        method_names = visualizer_autogluon.binary_methods
                        ambiguity_discrepancy_proba_plot, ambiguity_discrepancy_proba_descr = visualizer_autogluon.lolipop_ambiguity_discrepancy_proba_version(delta = delta)
                        plots_autogluon["lolipop_ambiguity_discrepancy_proba_version"], descriptions_autogluon["lolipop_ambiguity_discrepancy_proba_version"] = ambiguity_discrepancy_proba_plot, ambiguity_discrepancy_proba_descr
                        proba_ambiguity_plot, proba_ambiguity_descr = visualizer_autogluon.proba_ambiguity_vs_epsilon(delta = delta)
                        plots_autogluon["proba_ambiguity_vs_epsilon"], descriptions_autogluon["proba_ambiguity_vs_epsilon"] = proba_ambiguity_plot, proba_ambiguity_descr
                        proba_discrepancy_plot, proba_discrepancy_descr = visualizer_autogluon.proba_discrepancy_vs_epsilon(delta = delta)
                        plots_autogluon["proba_discrepancy_vs_epsilon"], descriptions_autogluon["proba_discrepancy_vs_epsilon"] = proba_discrepancy_plot, proba_discrepancy_descr
                    
                    elif rs_autogluon.task_type =="multiclass":
                        #autogluon plots
                        st.session_state.task_type = "multiclass"
                        method_names = visualizer_autogluon.multiclass_methods
                    
                    random_idx_autogluon = random.choice(y_true_autogluon.index.tolist()) #choose random sample for analysis
                    for method in method_names:
                            #autogluon
                            func = getattr(visualizer_autogluon, method)
                            sig = inspect.signature(func)
                            params = sig.parameters

                            if len(params)>0:
                                if "sample_index" in params:
                                    plot, descr = func(sample_index=random_idx_autogluon)
                                else: 
                                    raise ValueError(f"Method {method} needs unsupported parameters")
                            else: plot, descr = func()
                            plots_autogluon[method] = plot
                            descriptions_autogluon[method] = descr

                    spinner_placeholder.empty()
                    if st.session_state.task_type == "binary":
                        render_binary_dashboard(plots_autogluon, descriptions_autogluon, prefix="autogluon")
                    elif st.session_state.task_type == "multiclass":
                        render_multiclass_dashboard(plots_autogluon, descriptions_autogluon, prefix="autogluon") 
                except ValueError as e:
                    spinner_placeholder.empty()
                    st.warning("Please provide a greater epsilon value. For the selected base metric the Rashomon Set consists only of 1 model.")
            else:
                st.session_state.task_type = None
                
                if selected_dataset == "--choose--":
                    st.warning("Please select a dataset for analysis.")
                    
                elif selected_metric=="--choose--":
                    st.warning("Please choose evaluation metric.")
            
# h2o tab -------------------------------------------------------
        with h2o_tab:
            if all_params_set:
                leaderboard_h2o, y_true_h2o, predictions_dict_h2o, proba_predictions_dict_h2o, feature_importance_dict_h2o = load_converted_data(selected_dataset, "h2o")
                spinner_placeholder = st.empty()
                spinner_placeholder.markdown("""
                <div style="
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 200px;
                ">
                    <div style="
                        border: 4px solid #f3f3f3; 
                        border-top: 4px solid #426c85; 
                        border-radius: 50%;
                        width: 40px;
                        height: 40px;
                        animation: spin 1s linear infinite;
                    "></div>
                </div>

                <style>
                @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
                }
                </style>
                """, unsafe_allow_html=True)

                try:
                    rs_h2o = RashomonSet(leaderboard = leaderboard_h2o, predictions =predictions_dict_h2o, proba_predictions = proba_predictions_dict_h2o, feature_importances = feature_importance_dict_h2o, base_metric = selected_metric, epsilon = epsilon) 
                    visualizer_h2o = Visualizer(rs_h2o, y_true_h2o)
                    plots_h2o={}
                    descriptions_h2o={}
                    if rs_h2o.task_type=="binary":
                        method_names = visualizer_h2o.binary_methods
                        ambiguity_discrepancy_proba_plot, ambiguity_discrepancy_proba_descr = visualizer_h2o.lolipop_ambiguity_discrepancy_proba_version(delta = 0.1)
                        plots_h2o["lolipop_ambiguity_discrepancy_proba_version"], descriptions_h2o["lolipop_ambiguity_discrepancy_proba_version"] = ambiguity_discrepancy_proba_plot, ambiguity_discrepancy_proba_descr
                        proba_ambiguity_plot, proba_ambiguity_descr = visualizer_h2o.proba_ambiguity_vs_epsilon(delta = 0.1)
                        plots_h2o["proba_ambiguity_vs_epsilon"], descriptions_h2o["proba_ambiguity_vs_epsilon"] = proba_ambiguity_plot, proba_ambiguity_descr
                        proba_discrepancy_plot, proba_discrepancy_descr = visualizer_h2o.proba_discrepancy_vs_epsilon(delta = 0.1)
                        plots_h2o["proba_discrepancy_vs_epsilon"], descriptions_h2o["proba_discrepancy_vs_epsilon"] = proba_discrepancy_plot, proba_discrepancy_descr
                    elif rs_h2o.task_type == "multiclass":
                        st.session_state.task_type = "multiclass"
                        method_names = visualizer_h2o.multiclass_methods

                    random_idx_h2o = random.choice(y_true_h2o.index.tolist()) #choose random sample for analysis
                    for method in method_names:
                            #h2o
                            func = getattr(visualizer_h2o, method)
                            sig = inspect.signature(func)
                            params = sig.parameters

                            if len(params)>0:
                                if "sample_index" in params:
                                    plot, descr = func(sample_index=random_idx_h2o)
                                else: 
                                    raise ValueError(f"Method {method} needs unsupported parameters")
                            else: plot, descr = func()
                            plots_h2o[method] = plot
                            descriptions_h2o[method] = descr
                    spinner_placeholder.empty()
                    if st.session_state.task_type == "binary":
                        render_binary_dashboard(plots_h2o, descriptions_h2o, prefix="h2o")
                    elif st.session_state.task_type == "multiclass":
                        render_multiclass_dashboard(plots_h2o, descriptions_h2o, prefix="h2o") 
                except ValueError as e:
                    spinner_placeholder.empty()
                    st.warning("Please provide a greater epsilon value. For the selected base metric the Rashomon Set consists of only 1 model.")

            else:
                st.session_state.task_type = None
                
                if selected_dataset == "--choose--":
                    st.warning("Please select a dataset for analysis.")
                    
                elif selected_metric=="--choose--":
                    st.warning("Please choose evaluation metric.")





def visualize_rashomon_set_volume(leaderboard, predictions, proba_predictions, base_metric):
        '''
        Method for visualising Rashomon set size depending on epsilon
        '''
        epsilons = np.linspace(0, 1, 100)
        tmp_rashomon_set = RashomonSet(leaderboard, predictions, proba_predictions, None, base_metric, 10.0)
        valid_epsilons = [eps for eps in epsilons if len(tmp_rashomon_set.get_rashomon_set(eps))>1]

        def compute_size(epsilon):
            rs = RashomonSet(leaderboard, predictions, proba_predictions, None, base_metric, epsilon)
            return len(rs.rashomon_set)
            
        rashomon_sizes= [compute_size(eps) for eps in valid_epsilons]
        # jump epsilons
        diff = np.diff(rashomon_sizes)
        jump_indices = np.where(diff > 0)[0] + 1

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(valid_epsilons, rashomon_sizes, s=20, color='#16476A')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        for idx in jump_indices:
            jump_epsilon = valid_epsilons[idx]
            jump_epsilon = valid_epsilons[idx]
            ax.axvline(x=jump_epsilon, color='#eea951', linestyle='--', alpha=0.8)
            ax.text(jump_epsilon, ax.get_ylim()[0] - 0.2, f"{jump_epsilon:.3f}",
                rotation=90, color='#eea951', fontsize=8, ha='center', va='top')

        if len(jump_indices) > 0:
            ax.axvline(x=valid_epsilons[jump_indices[0]], color='#eea951', linestyle='--', alpha=0.5,
                        label='Change in the Rashomon set size')
        ax.legend(fontsize=10)
        ax.set_xlabel('Epsilon value', labelpad=19)
        ax.set_ylabel('Number of models in Rashomon set')
        ax.set_title('Rashomon set sizes', pad=10)
        plt.tight_layout()
        return fig








       
           

            
            
        
        
            
        
        


       



        
  
    
        
        
    