import os
import streamlit as st

"""
# Grades
Behind the Secrets of Large Language Models
"""

email = st.text_input("Email", placeholder="Enter your university email")

st.text(os.environ.get(f"grades.{email}"))