import streamlit as st
from pathlib import Path
from other_pages import rashomon_page, intersection_page
current_dir = Path(__file__).parent
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