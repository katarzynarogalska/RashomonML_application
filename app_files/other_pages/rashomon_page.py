import streamlit as st

def show():
    st.title("Rashomon Page")
    
    if st.session_state.global_dataset is None:
        st.warning("Please select a dataset for analysis. You may choose a dataset from the list on the sidebar.")
    elif st.session_state.base_metric is None or st.session_state.epsilon is None:
        st.warning("Please expand the 'Specify parameters' section in order to provide the necessary parameters, such as the base metric and epsilon")
    
    else:
        st.markdown("SELECTED DATASET:")
        st.markdown(st.session_state.global_dataset)
        st.markdown("Selected metric")
        st.markdown(st.session_state.base_metric)
        st.markdown("Selected epsilon")
        st.markdown(st.session_state.epsilon)
    
    
    