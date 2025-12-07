import streamlit as st

def show():
    st.title("Rashomon Page")
    selected_dataset = st.session_state.get('global_dataset', "")
    if selected_dataset =="":
        st.warning("Please select a dataset for analysis")
    else:
        st.markdown("SELECTED DATASET:")
        st.markdown(selected_dataset)
    
    
    