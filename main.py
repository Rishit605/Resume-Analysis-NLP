import streamlit as st
import pandas as pd
import numpy as np

st.header("Resume Analysis and Classification App")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
	st.write("File Not found! Please Try Again!")