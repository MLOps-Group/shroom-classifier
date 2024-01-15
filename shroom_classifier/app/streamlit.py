import streamlit as st

### Main functionality
with st.sidebar:
    st.title("Shroom_classifier")
    
    st.text("Upload Image of a mushroom to classify it")
    video_file = st.file_uploader("Upload Image", type=["jpg"])

    
    # if implementation == "Demo":
    #   demo()