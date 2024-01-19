import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import plotly.express as px
import os

url = os.environ.get("URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="Shroom_classifier", page_icon="🍄", layout="wide")

### Main functionality
with st.sidebar:
    st.title("Shroom classifier")

    st.text('''Upload Image of a mushroom
to classify it''')
    way = st.radio("Choose a way to upload image", ["Upload from device", "Use Camera"])
    if way == "Upload from device":
        img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    else:
        img_file_buffer = st.camera_input("Take a picture")
        img = None
        if img_file_buffer is not None:
            img = img_file_buffer

    #img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if img is not None:
        st.image(img, caption="What a beautiful Shroom!")

st.title("🍄Know Your Shroom: A Mushroom classifier🍄")
if img is not None:
    left_co, last_co = st.columns(2)
    with left_co:
        st.write("Predicting...")
        response = requests.post(f"{url}/predict", files={"file": img})
        if response.status_code == 200:
            st.write("Done!")

            #table from jsonfile
            data = response.json()
            data = data["top_k_preds"]
            probs = data["probs"][0]
            labels = data["labels"]

            df = pd.DataFrame({"Probability": probs, "Label": labels})
            df = df.set_index("Label")
            df = df.sort_values(by="Probability", ascending=True)

            fig = px.bar(df, x="Probability", y=df.index, title="Probability of each class", orientation="h")

            #make bar chart
            st.plotly_chart(fig, use_container_width=True)
        with last_co:
            try:
                iframe_src = "https://en.wikipedia.org/wiki/{0}".format(labels[0].replace(" ", "_"))
                components.iframe(iframe_src, height=800, scrolling=True)
            except Exception:
                st.write('''Could not find wikipedia page for this mushroom, sorry!
                         Here is the wikipedia page for mushrooms in general''')
                components.iframe("https://en.wikipedia.org/wiki/Mushroom", height=800, scrolling=True)
