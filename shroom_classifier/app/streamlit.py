import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import plotly.express as px

url = "http://127.0.0.1:8000"
st.set_page_config(page_title="Shroom_classifier", page_icon="🍄", layout="wide")

### Main functionality
with st.sidebar:
    st.title("Shroom classifier")
    
    st.text('''Upload Image of a mushroom 
to classify it''')
    img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

st.title("Know Your Shroom: A Mushroom classifier")
if img is not None:
    left_co, last_co = st.columns(2)
    with left_co:
        st.image(img, caption="What a beautiful Shroom!", use_column_width=True)
    with last_co:
        st.write("Predicting...")
        response = requests.post(f"{url}/predict", files={"file": img})
        if response.status_code == 200:
            st.write("Done!")

            #table from jsonfile
            data = response.json()
            data = data["top_k_preds"]
            probs = data["probs"][0]
            labels = data["labels"]
            
            print(len(probs))
            print(len(labels))

            df = pd.DataFrame({"Probability": probs, "Label": labels})
            df = df.set_index("Label")
            df = df.sort_values(by="Probability", ascending=True)

            fig = px.bar(df, x="Probability", y=df.index, title="Probability of each class", orientation="h")

            #make bar chart
            st.plotly_chart(fig, use_container_width=True)

    iframe_src = "https://en.wikipedia.org/wiki/{0}".format(labels[0].replace(" ", "_"))
    components.iframe(iframe_src, height=800, scrolling=True)
    

