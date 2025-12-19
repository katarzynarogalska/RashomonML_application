import streamlit as st
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

def show():

        st.markdown('<div class="homepage_title"> &nbsp; The Rashomon Set Analysis </div>', unsafe_allow_html=True)
        st.markdown("")
        with st.container(key="params"):
            st.markdown('<div class="params_title"> &nbsp; Specify the necessary parameters : </div>', unsafe_allow_html=True)
  
            col0,col1,col11, col2, col22, col3, col4 = st.columns([0.2,1,0.5,1,0.5,1,0.2])
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


        autogluon_tab, h2o_tab = st.tabs(["AutoGluon", "H2O"])
        with autogluon_tab:
            if selected_dataset == "--choose--":
                st.warning("Please select a dataset for analysis")
            else:
                if selected_metric =="--choose--":
                    st.warning("Please choose a base metric")
            if selected_dataset!="--choose--" and selected_metric!="--choose--":
                st.markdown("Autogluon analysis")
                st.markdown("SELECTED DATASET:")
                st.markdown(selected_dataset)
                st.markdown("Selected metric")
                st.markdown(selected_metric)
                st.markdown("Selected epsilon")
                st.markdown(epsilon)

        with h2o_tab:
            if selected_dataset == "--choose--":
                st.warning("Please select a dataset for analysis")
            else:
                if selected_metric =="--choose--":
                    st.warning("Please choose a base metric")
            if selected_dataset!="--choose--":
                st.markdown("H2o analysis")
                st.markdown("SELECTED DATASET:")
                st.markdown(selected_dataset)
                st.markdown("Selected metric")
                st.markdown(selected_metric)
                st.markdown("Selected epsilon")
                st.markdown(epsilon)
        
  
    
        
        
    