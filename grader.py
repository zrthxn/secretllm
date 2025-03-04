import os
import streamlit as st

"""
# Grades
Behind the Secrets of Large Language Models
"""

matric = st.text_input("Matriculation Number", placeholder="Enter your Matriculation Number")

labels = {
    "1.0": "Very Good",
    "1.3": "Very Good",
    "1.7": "Very Good",
    "2.0": "Good",
    "2.3": "Good",
    "2.7": "Good",
    "3.0": "Satisfactory",
    "3.3": "Satisfactory",
    "3.7": "Sufficient",
    "4.0": "Sufficient",
    "5.0": "Failed",
}

if matric:
    grade = getattr(st.secrets, f'{matric}')
    label = labels[grade]
    st.markdown(
    f"""
    <div style="text-align: center; padding: 1em;">
        <h1 style="padding: 0;">{grade}</h1>
        <h3>{label}</h3>
    </div>
    """,
    unsafe_allow_html=True)
else:
    st.markdown("""<br><br>""", unsafe_allow_html=True)

st.markdown(
"""
<div style="text-align: center; color: gray;">
Please note this grade is only provided as a preview for your information. 
<br>
For Komplexprüfungen, as decided, your current grade will be carried over to the oral exam.

<br><br>
For greviances, we can provide a more detailed breakdown of your grade.
<br>
Please write to syed_alisamar.husain@tu-dresden.de
</div>
""", 
unsafe_allow_html=True)
