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
        with autogluon_tab:
            if selected_dataset == "--choose--":
                st.warning("Please select a dataset for analysis")
            
            elif st.session_state.metric1 =="--choose--" or st.session_state.metric2 =="--choose--":
                st.warning("Please choose two evaluation metrics")
            else:
                st.markdown("Autogluon analysis")
                st.markdown("SELECTED DATASET:")
                st.markdown(selected_dataset)
                st.markdown("Selected metric1")
                st.markdown(st.session_state.metric1)
                st.markdown("Selected metric2")
                st.markdown(st.session_state.metric2)
                st.markdown("Selected epsilon")
                st.markdown(epsilon)
                st.markdown("Selected method")
                st.markdown(selected_sum_method)

        with h2o_tab:
            if selected_dataset == "--choose--":
                st.warning("Please select a dataset for analysis")
            
            elif st.session_state.metric1 =="--choose--" or st.session_state.metric2 =="--choose--":
                st.warning("Please choose two evaluation metrics")
            else:
                st.markdown("H2o analysis")
                st.markdown("SELECTED DATASET:")
                st.markdown(selected_dataset)
                st.markdown("Selected metric1")
                st.markdown(st.session_state.metric1)
                st.markdown("Selected metric2")
                st.markdown(st.session_state.metric2)
                st.markdown("Selected epsilon")
                st.markdown(epsilon)
                st.markdown("Selected method")
                st.markdown(selected_sum_method)