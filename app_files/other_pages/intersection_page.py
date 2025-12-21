import streamlit as st
import pandas as pd
import inspect
from rashomon_analysis.rashomon_intersection import RashomonIntersection
from rashomon_analysis.visualizers.intersection_visualizer import IntersectionVisualizer
from .dashboard.intersection_binary import render_binary_dashboard_intersection
from .dashboard.intersection_multiclass import render_multiclass_dashboard_intersection
import pickle
from pathlib import Path
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

        st.markdown('<div class="homepage_title"> &nbsp; The Rashomon Intersection Analysis </div>', unsafe_allow_html=True)
        st.markdown("")
        with st.container(key="params"):
            st.markdown('<div class="params_title"> &nbsp; Specify the necessary parameters : </div>', unsafe_allow_html=True)
  
            col0,col1, col2, col3, col4,col5, col6 = st.columns([0.2,1,1,1,1,1,0.2])
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
                    
                    st.session_state.metric1= st.selectbox(
                    "First metric:",
                    options=metrics_options,
                    index=0,  
                    key="base_metric_select"
                    )
                    if st.session_state.metric1!="--choose--":
                        st.markdown(f'<div class="selected_params"> Selected first metric: {st.session_state.metric1} </div>', unsafe_allow_html=True)

                    metrics_options2 = ["--choose--"] + [m for m in metrics_options if m != st.session_state.metric1 and m != "--choose--"]
                    st.session_state.metric2= st.selectbox(
                    "Second metric:",
                    options=metrics_options2,
                    index=0
                    )
                    if st.session_state.metric2!="--choose--":
                        st.markdown(f'<div class="selected_params"> Selected second metric: {st.session_state.metric2} </div>', unsafe_allow_html=True)

            with col3:
                if selected_dataset!="--choose--":
                    epsilon = st.slider("Epsilon:", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
                    st.markdown(f'<div class="selected_params"> Selected epsilon : {epsilon} </div>', unsafe_allow_html=True)

            with col4:
                if selected_dataset!="--choose--":
                    selected_sum_method = st.radio("Weighted sum method:", [ "entropy","critic","custom_weights"])
            with col5:
                if selected_dataset!="--choose--":
                    if selected_sum_method=="custom_weights":
                        if st.session_state.metric1=="--choose--" or st.session_state.metric2=="--choose--":
                            st.markdown("Please select metrics first.")
                        else:
                            st.session_state.weight1 = st.number_input(f"Weight for {st.session_state.metric1 or 'Metric 1'}:", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="weight1_input")
                
                            st.session_state.weight2 = 1.0 - st.session_state.weight1
                            st.number_input(f"Weight for {st.session_state.metric2 or 'Metric 2'}:", min_value=0.0, max_value=1.0, value=st.session_state.weight2, step=0.01, key="weight2_input", disabled=True)

        autogluon_tab, h2o_tab = st.tabs(["AutoGluon", "H2O"])
        all_params_set = (selected_dataset != "--choose--" and st.session_state.metric1 != "--choose--" and st.session_state.metric2 != "--choose--" and selected_sum_method is not None)
        if all_params_set:
            leaderboard_autogluon, y_true_autogluon, predictions_dict_autogluon, proba_predictions_dict_autogluon, feature_importance_dict_autogluon = load_converted_data(selected_dataset, "autogluon")
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

            rs_autogluon = RashomonIntersection(leaderboard = leaderboard_autogluon, predictions = predictions_dict_autogluon, proba_predictions = proba_predictions_dict_autogluon, feature_importances = feature_importance_dict_autogluon, metrics = [st.session_state.metric1, st.session_state.metric2], epsilon = epsilon, custom_weights = ( [st.session_state.weight1, st.session_state.weight2] if selected_sum_method=="custom_weights" else None), weighted_sum_method = selected_sum_method)
            rs_h2o = RashomonIntersection(leaderboard = leaderboard_h2o, predictions = predictions_dict_h2o, proba_predictions = proba_predictions_dict_h2o, feature_importances = feature_importance_dict_h2o, metrics = [st.session_state.metric1, st.session_state.metric2], epsilon = epsilon, custom_weights = ( [st.session_state.weight1, st.session_state.weight2] if selected_sum_method=="custom_weights" else None), weighted_sum_method = selected_sum_method)
            visualizer_autogluon = IntersectionVisualizer(rs_autogluon, y_true_autogluon)
            visualizer_h2o = IntersectionVisualizer(rs_h2o, y_true_h2o)

            plots_autogluon, plots_h2o ={}, {}
            descriptions_autogluon, descriptions_h2o ={}, {}

            if rs_autogluon.task_type == "binary":
                #autoglon plots
                st.session_state.task_type = "binary"
                method_names = visualizer_autogluon.binary_methods
                ambiguity_discrepancy_proba_plot, ambiguity_discrepancy_proba_descr = visualizer_autogluon.lolipop_ambiguity_discrepancy_proba_version(delta = 0.1)
                plots_autogluon["lolipop_ambiguity_discrepancy_proba_version"], descriptions_autogluon["lolipop_ambiguity_discrepancy_proba_version"] = ambiguity_discrepancy_proba_plot, ambiguity_discrepancy_proba_descr
                proba_ambiguity_plot, proba_ambiguity_descr = visualizer_autogluon.proba_ambiguity_vs_epsilon(delta = 0.1)
                plots_autogluon["proba_ambiguity_vs_epsilon"], descriptions_autogluon["proba_ambiguity_vs_epsilon"] = proba_ambiguity_plot, proba_ambiguity_descr
                proba_discrepancy_plot, proba_discrepancy_descr = visualizer_autogluon.proba_discrepancy_vs_epsilon(delta = 0.1)
                plots_autogluon["proba_discrepancy_vs_epsilon"], descriptions_autogluon["proba_discrepancy_vs_epsilon"] = proba_discrepancy_plot, proba_discrepancy_descr
                #h2o plots
                method_names = visualizer_h2o.binary_methods
                ambiguity_discrepancy_proba_plot, ambiguity_discrepancy_proba_descr = visualizer_h2o.lolipop_ambiguity_discrepancy_proba_version(delta = 0.1)
                plots_h2o["lolipop_ambiguity_discrepancy_proba_version"], descriptions_h2o["lolipop_ambiguity_discrepancy_proba_version"] = ambiguity_discrepancy_proba_plot, ambiguity_discrepancy_proba_descr
                proba_ambiguity_plot, proba_ambiguity_descr = visualizer_h2o.proba_ambiguity_vs_epsilon(delta = 0.1)
                plots_h2o["proba_ambiguity_vs_epsilon"], descriptions_h2o["proba_ambiguity_vs_epsilon"] = proba_ambiguity_plot, proba_ambiguity_descr
                proba_discrepancy_plot, proba_discrepancy_descr = visualizer_h2o.proba_discrepancy_vs_epsilon(delta = 0.1)
                plots_h2o["proba_discrepancy_vs_epsilon"], descriptions_h2o["proba_discrepancy_vs_epsilon"] = proba_discrepancy_plot, proba_discrepancy_descr

            elif rs_autogluon.task_type =="multiclass":
                #autogluon plots
                st.session_state.task_type = "multiclass"
                method_names = visualizer_autogluon.multiclass_methods
            random_idx_autogluon = random.choice(y_true_autogluon.index.tolist()) #choose random sample for analysis
            random_idx_h2o = random.choice(y_true_h2o.index.tolist()) #choose random sample for analysis
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
        else:
            st.session_state.task_type = None
            if selected_dataset == "--choose--":
                st.warning("Please select a dataset for analysis.")
            
            elif st.session_state.metric1 =="--choose--" or st.session_state.metric2 =="--choose--":
                st.warning("Please choose two evaluation metrics.")
                
        with autogluon_tab:
            if all_params_set:
                if st.session_state.task_type == "binary":
                    render_binary_dashboard_intersection(plots_autogluon, descriptions_autogluon, prefix="autogluon")
                elif st.session_state.task_type == "multiclass":
                    render_multiclass_dashboard_intersection(plots_autogluon, descriptions_autogluon, prefix="autogluon")
                                

        with h2o_tab:
            if all_params_set:
                if st.session_state.task_type == "binary":
                    render_binary_dashboard_intersection(plots_h2o, descriptions_h2o, prefix="h2o")
                elif st.session_state.task_type == "multiclass":
                    render_multiclass_dashboard_intersection(plots_h2o, descriptions_h2o, prefix="h2o")