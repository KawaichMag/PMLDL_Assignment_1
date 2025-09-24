import os

import requests
import streamlit as st

st.set_page_config(page_title="Titanic Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.caption("Enter your own values to check if you survive")
st.info(
    "Family size category: Single (1 person), Couple (2), InterM (3-4), Large (5+).\n"
)

api_url = os.getenv("API_BASE_URL", "http://localhost:8000")

col1, col2 = st.columns(2)
with col1:
    p_class = st.selectbox(
        "Passenger class (Pclass)",
        [1, 2, 3],
        index=2,
        help="Ticket class: 1=Upper, 2=Middle, 3=Lower",
    )
    sex = st.selectbox(
        "Sex",
        ["male", "female"],
        index=0,
        help="Biological sex as recorded on ticket (male/female)",
    )
    embarked = st.selectbox(
        "Embarked",
        ["S", "Q", "C"],
        index=0,
        help="Boarding port: S=Southampton, Q=Queenstown, C=Cherbourg",
    )
    title = st.selectbox(
        "Title",
        ["Master", "Mr", "Mrs", "Miss"],
        index=0,
        help="Honorific title on the ticket (use closest)",
    )
with col2:
    age = st.number_input(
        "Age",
        min_value=0,
        max_value=100,
        value=30,
        step=1,
        help="Age in years",
    )
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=32.2,
        step=0.1,
        help="Ticket fare (dataset units; approx. British pounds)",
    )
    family_size = st.selectbox(
        "Family size",
        ["Single", "Couple", "InterM", "Large"],
        index=0,
        help="Choose by total people aboard in your family group: 1→Single, 2→Couple, 3-4→InterM, 5+→Large",
    )

st.divider()

predict_clicked = st.button("Predict")

if "show_modal" not in st.session_state:
    st.session_state.show_modal = False
if "modal_text" not in st.session_state:
    st.session_state.modal_text = ""

if predict_clicked:
    try:
        params = {
            "p_class": int(p_class),
            "sex": sex,
            "age": int(age),
            "fare": float(fare),
            "embarked": embarked,
            "title": title,
            "family_size": family_size,
        }
        resp = requests.get(f"{api_url}/predict_me", params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        label_str = payload.get("result", "0")
        try:
            label = int(label_str)
        except Exception:
            label = 0
        st.session_state.modal_text = (
            "You survived!" if label == 1 else "You not survived.."
        )
        st.session_state.show_modal = True
        st.toast("Prediction received")
    except Exception as e:
        st.session_state.modal_text = f"Request failed: {e}"
        st.session_state.show_modal = True
        st.toast("Prediction failed")

if st.session_state.show_modal:
    if st.session_state.modal_text == "You survived!":
        st.success(st.session_state.modal_text)
    else:
        st.error(st.session_state.modal_text)
    st.button("Close", on_click=lambda: st.session_state.update({"show_modal": False}))
