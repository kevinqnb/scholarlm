import streamlit as st
import json
import pandas as pd # Optional, for displaying results nicely
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

####################################################################################################
# --- Load Data ---

filename = "../data/pond_results4.csv"
outfile = "../data/pond_results4_validated.csv"

n_sample = 100

# Random sample for validation:
df = pd.read_csv(filename)
ignore_measurements = ['latitude', 'longitude'] # Ignoring these because they are not in the original dataset
df = df.loc[~df.measurement.isin(ignore_measurements)]
df = df.sample(n=n_sample, random_state=42)
df = df.sort_values(by=['title', 'chunk_id'])
df = df.reset_index(drop=True)

# Isolate data points and their corresponding markdown content
context_score_cols = [col for col in df.columns if col.startswith('context_')]
parametric_score_cols = [col for col in df.columns if col.startswith('parametric_')]
data_df = df.drop(columns = context_score_cols + parametric_score_cols + ['context', 'chunk_id'])
data_points = data_df.to_dict(orient='records')
data_points = [{k: v for k, v in d.items()} for d in data_points]
markdown_content = df.loc[:, 'context'].tolist()

# Combine markdown and data into a single list of dictionaries
sample_dataset = [
    {"markdown_text": md, "data": data}
    for md, data in zip(markdown_content, data_points)
]


####################################################################################################
# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="Dataset Validator")

st.title("Dataset Validation")
st.write("Review each data point and its associated Markdown.")

# Initialize session state for storing results and current index
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = []
if 'identification_results' not in st.session_state:
    st.session_state.identification_results = []
if 'unit_results' not in st.session_state:
    st.session_state.unit_results = []
if 'dataset' not in st.session_state:
    st.session_state.dataset = sample_dataset

dataset_len = len(st.session_state.dataset)

def record_validation(status):
    """Records the validation status for the current data point and advances to next."""
    # Ensure we have responses for all questions before advancing
    current_idx = st.session_state.current_index
    if (len(st.session_state.identification_results) > current_idx and 
        len(st.session_state.unit_results) > current_idx):
        st.session_state.validation_results.append(status)
        st.session_state.current_index += 1
    else:
        st.error("Please answer Q1 and Q2 before proceeding to the next data point.")

def record_identification(identification_status):
    """Records the identification result for the current data point."""
    current_idx = st.session_state.current_index
    # Ensure we don't record multiple responses for the same data point
    if len(st.session_state.identification_results) <= current_idx:
        st.session_state.identification_results.append(identification_status)
    else:
        st.session_state.identification_results[current_idx] = identification_status

def record_unit(unit_status):
    """Records the unit correctness for the current data point."""
    current_idx = st.session_state.current_index
    # Ensure we don't record multiple responses for the same data point
    if len(st.session_state.unit_results) <= current_idx:
        st.session_state.unit_results.append(unit_status)
    else:
        st.session_state.unit_results[current_idx] = unit_status

def save_results():
    """Saves the validation results to CSV file."""
    # Ensure all result lists have the correct length
    n_results = len(st.session_state.validation_results)
    if (len(st.session_state.identification_results) == n_results and 
        len(st.session_state.unit_results) == n_results and
        n_results <= len(df)):
        
        # Create a copy of the dataframe with only the validated rows
        validated_df = df.iloc[:n_results].copy()
        validated_df['identification_status'] = st.session_state.identification_results
        validated_df['unit_status'] = st.session_state.unit_results
        validated_df['validation_status'] = st.session_state.validation_results
        validated_df.to_csv(outfile, index=False)
    else:
        st.error("Error: Mismatch in result lengths. Cannot save results.")


# --- Display Current Data Point ---
if st.session_state.current_index < dataset_len:
    current_item = st.session_state.dataset[st.session_state.current_index]

    st.subheader(f"Data Point {st.session_state.current_index + 1} of {dataset_len}")

    st.write("### Q1: Is this data point identified correctly?")
    current_idx = st.session_state.current_index
    q1_answered = len(st.session_state.identification_results) > current_idx
    
    col_id1, col_id2 = st.columns(2)
    with col_id1:
        if st.button("Yes", key="id_yes_btn", disabled=q1_answered):
            record_identification(True)
            st.rerun()
    with col_id2:
        if st.button("No", key="id_no_btn", disabled=q1_answered):
            record_identification(False)
            st.rerun()
    
    if q1_answered:
        st.success(f"✓ Answered: {'Yes' if st.session_state.identification_results[current_idx] else 'No'}")

    st.markdown("---")
    st.write("### Q2: Are the units correct for this data point?")
    q2_answered = len(st.session_state.unit_results) > current_idx
    
    col_unit1, col_unit2 = st.columns(2)
    with col_unit1:
        if st.button("Yes", key="unit_yes_btn", disabled=q2_answered):
            record_unit(True)
            st.rerun()
    with col_unit2:
        if st.button("No", key="unit_no_btn", disabled=q2_answered):
            record_unit(False)
            st.rerun()
            
    if q2_answered:
        st.success(f"✓ Answered: {'Yes' if st.session_state.unit_results[current_idx] else 'No'}")

    st.markdown("---")

    st.write("### Q3: Is this data point valid?")
    # Only show Q3 if Q1 and Q2 are answered
    if q1_answered and q2_answered:
        col_btns1, col_btns2, col_btns3, col_btns4, col_btns5 = st.columns(5)
        with col_btns1:
            if st.button("Valid", key="valid_btn"):
                record_validation("valid")
                st.rerun()
        with col_btns2:
            if st.button("Hallucination", key="hallucination_btn"):
                record_validation("hallucination")
                st.rerun()
        with col_btns3:
            if st.button("Disorientation", key="disorientation_btn"):
                record_validation("disorientation")
                st.rerun()
        with col_btns4:
            if st.button("Deviation", key="deviation_btn"):
                record_validation('deviation')
                st.rerun()
        with col_btns5:
            if st.button("Skip", key="skip_btn"):
                record_validation(None)
                st.rerun()
    else:
        st.info("Please answer Q1 and Q2 first before proceeding to Q3.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.warning("Markdown Text:")
        st.markdown(current_item['markdown_text'])

    with col2:
        st.info("Extracted Data:")
        st.json(current_item['data'])

    st.markdown("---")


else:
    st.success("All data points have been reviewed!")
    st.write("### Summary of Validation Results")
    st.write(f"Total Data Points Reviewed: {len(st.session_state.validation_results)}")
    st.write(f"✅ Valid: {st.session_state.validation_results.count('valid')}")
    st.write(f"❌ Hallucinations: {st.session_state.validation_results.count('hallucination')}")
    st.write(f"❌ Disorientations: {st.session_state.validation_results.count('disorientation')}")
    st.write(f"❌ Deviations: {st.session_state.validation_results.count('deviation')}")
    st.write(f"⏭️ Skipped: {st.session_state.validation_results.count(None)}")

    if st.session_state.validation_results:
        # save to CSV
        save_results()
        st.write(f"Validation results saved to `{outfile}`.")
    else:
        st.write("No validation decisions were made.")

    if st.button("Start Over"):
        st.session_state.current_index = 0
        st.session_state.validation_results = []
        st.session_state.identification_results = []
        st.session_state.unit_results = []
        st.rerun()

####################################################################################################