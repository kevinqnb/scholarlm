import streamlit as st
import json
import pandas as pd # Optional, for displaying results nicely
import numpy as np
import os
import random
from dotenv import load_dotenv
load_dotenv()
from pdf2image import convert_from_path
from scholarlm.utils import get_foldernames_in_directory

main_directory = os.getenv("POND_PATH")
image_directory = os.getenv("POND_IMAGE_PATH")
image_folders = get_foldernames_in_directory(image_directory, ignore = [".DS_Store"])
pdf_directory = os.getenv("POND_PDF_PATH")
image_folders.sort()

####################################################################################################
# --- Cached Functions ---

@st.cache_data
def load_pdf_page(pdf_path, page_id):
    """Load a specific page from a PDF as an image. Cached to avoid reprocessing."""
    # Convert only the specific page we need (pages are 1-indexed in pdf2image)
    try:
        images = convert_from_path(pdf_path, dpi=100, first_page=page_id+1, last_page=page_id+1)
        return images[0]
    except Exception as e:
        st.error(f"Error loading page {page_id} from {pdf_path}: {e}")
        return None


@st.cache_data
def load_dataset(filename, n_sample):
    """Load and prepare the dataset. Cached to avoid reloading on every interaction."""
    with open(filename, "r") as f:
        result_dict = json.load(f)
    
    #ignore_measurements = ['latitude', 'longitude'] # Ignoring these because they are not in the original dataset
    
    # Get points which are not already matched to ground truth:
    #result_dict = [entry for entry in result_dict if entry['matched'] == False]
    #result_dict = [{'units': None} | entry for entry in result_dict]
    ignore_measurements = ['latitude', 'longitude']
    result_dict = [entry for entry in result_dict if entry['measurement'] not in ignore_measurements]
    
    random.seed(42)
    result_dict_sample = random.sample(result_dict, k=n_sample)
    result_dict_sample = sorted(result_dict_sample, key=lambda x: (x['document_id'], x['page_id']))
    
    #drops = ['context', 'context_scores', 'parametric_scores', 'copying_scores', 'linear_probes']
    display = ['paper_code', 'document_id', 'page_id', 'title', 'author', 'year', 'name', 'location', 'date', 'ecosystem', 'measurement', 'value', 'units', 'judgement']
    result_dict_data_points = [{feature: entry.get(feature, None) for feature in display} for entry in result_dict_sample]
    markdown_content = [entry['context'] for entry in result_dict_sample]
    
    # Combine markdown and data into a single list of dictionaries
    sample_dataset = [
        {"markdown_text": md, "data": data}
        for md, data in zip(markdown_content, result_dict_data_points)
    ]
    
    return sample_dataset, result_dict_sample


####################################################################################################
# --- Load Data ---

filename = "../data/12_11_25/pond.json"
outfile = "../data/12_11_25/pond_validated.json"

n_sample = 100

# Load dataset using cached function
sample_dataset, result_dict_sample = load_dataset(filename, n_sample)


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
if 'unit_results' not in st.session_state:
    st.session_state.unit_results = []
if 'ocr_results' not in st.session_state:
    st.session_state.ocr_results = []
if 'dataset' not in st.session_state:
    st.session_state.dataset = sample_dataset
if 'stopped_early' not in st.session_state:
    st.session_state.stopped_early = False

dataset_len = len(st.session_state.dataset)


def record_ocr_status(ocr_status):
    """Records the unit correctness for the current data point."""
    current_idx = st.session_state.current_index
    # Ensure we don't record multiple responses for the same data point
    if len(st.session_state.ocr_results) <= current_idx:
        st.session_state.ocr_results.append(ocr_status)
    else:
        st.session_state.ocr_results[current_idx] = ocr_status


def record_unit_status(unit_status):
    """Records the unit correctness for the current data point."""
    current_idx = st.session_state.current_index
    # Ensure we don't record multiple responses for the same data point
    if len(st.session_state.unit_results) <= current_idx:
        st.session_state.unit_results.append(unit_status)
    else:
        st.session_state.unit_results[current_idx] = unit_status


def record_validation_status(validation_status):
    """Records the validation status for the current data point and advances to next."""
    # Ensure we have responses for all questions before advancing
    current_idx = st.session_state.current_index
    if (len(st.session_state.validation_results) <= current_idx):
        st.session_state.validation_results.append(validation_status)
        st.session_state.current_index += 1
    else:
        st.error("Please answer previous question(s) before proceeding.")


def save_results():
    """Saves the validation results to CSV file."""
    # Ensure all result lists have the correct length
    n_results = len(st.session_state.validation_results)
    if (len(st.session_state.unit_results) == n_results and
        len(st.session_state.ocr_results) == n_results and
        n_results <= len(result_dict_sample)):
        
        # Create a copy of the dataframe with only the validated rows
        validated_data = []
        for i in range(n_results):
            datapoint = result_dict_sample[i]
            datapoint['unit_status'] = st.session_state.unit_results[i]
            datapoint['ocr_status'] = st.session_state.ocr_results[i]
            datapoint['validation_status'] = st.session_state.validation_results[i]
            validated_data.append(datapoint)
        
        with open(outfile, 'w') as f:
            json.dump(validated_data, f, indent=4)
    else:
        st.error("Error: Mismatch in result lengths. Cannot save results.")


# --- Display Current Data Point ---
if st.session_state.current_index < dataset_len and not st.session_state.stopped_early:
    current_item = st.session_state.dataset[st.session_state.current_index]
    paper_code = current_item['data']['paper_code']
    document_id = current_item['data']['document_id']
    page_id = int(current_item['data']['page_id'])
    #chunk_id = current_item['data']['chunk_id']
    
    #image_path = os.path.join(image_directory, image_folders[document_id], f"chunk_{chunk_id}.png")
    pdf_path = os.path.join(pdf_directory, f"{paper_code}.pdf")
    
    # Load the PDF page using the cached function - this will only convert once per page
    current_img = load_pdf_page(pdf_path, page_id)

    st.subheader(f"Data Point {st.session_state.current_index + 1} of {dataset_len}")
    
    # Add "Stop and Save" button at the top
    if st.button("🛑 Stop and Save Results", key="stop_btn", type="secondary"):
        st.session_state.stopped_early = True
        st.rerun()

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.warning("Image:")
        if current_img is not None:
            st.image(current_img, caption=f"Paper: {image_folders[document_id]}, Page ID: {page_id}")
        #st.image(image_path, caption=f"Paper: {image_folders[document_id]}, Chunk ID: {chunk_id}")#, use_column_width=True)
        st.warning("Markdown Text:")
        st.markdown(current_item['markdown_text'], unsafe_allow_html=True)

    with col2:
        st.info("Extracted Data:")
        st.json(current_item['data'])

    current_idx = st.session_state.current_index
    st.markdown("---")
    st.write("### Q1: Does the OCR text correctly represent the pdf well enough to be able to extract data for the given entity?")
    q1_answered = len(st.session_state.ocr_results) > current_idx
    
    col_unit1, col_unit2, col_unit3 = st.columns(3)
    with col_unit1:
        if st.button("Yes", key="ocr_yes_btn", disabled=q1_answered):
            record_ocr_status(True)
            st.rerun()
    with col_unit2:
        if st.button("No", key="ocr_no_btn", disabled=q1_answered):
            record_ocr_status(False)
            st.rerun()
    with col_unit3:
        if st.button("N/A", key="ocr_na_btn", disabled=q1_answered):
            record_ocr_status(None)
            st.rerun()
            
    if q1_answered:
        st.success(f"✓ Answered: {'Yes' if st.session_state.ocr_results[current_idx] else 'No'}")

    st.markdown("---")

    st.write("### Q2: With respect to the OCR generated text, are the units correct for this data point's value?")
    q2_answered = len(st.session_state.unit_results) > current_idx
    
    if q1_answered:
        col_unit1, col_unit2, col_unit3 = st.columns(3)
        with col_unit1:
            if st.button("Yes", key="unit_yes_btn", disabled=q2_answered):
                record_unit_status(True)
                st.rerun()
        with col_unit2:
            if st.button("No", key="unit_no_btn", disabled=q2_answered):
                record_unit_status(False)
                st.rerun()
        with col_unit3:
            if st.button("N/A", key="unit_na_btn", disabled=q2_answered):
                record_unit_status(None)
                st.rerun()
    else:
        st.info("Please answer previous question before proceeding.")
                
    if q2_answered:
        st.success(f"✓ Answered: {'Yes' if st.session_state.unit_results[current_idx] else 'No'}")

    st.markdown("---")

    st.write("### Q3: With respect to the OCR generated text, how should this data point's value be classified?")
    if q2_answered:
        col_btns1, col_btns2, col_btns3, col_btns4, col_btns5 = st.columns(5)
        with col_btns1:
            if st.button("Valid", key="valid_btn"):
                record_validation_status("valid")
                st.rerun()
        with col_btns2:
            if st.button("Hallucination", key="hallucination_btn"):
                record_validation_status("hallucination")
                st.rerun()
        with col_btns3:
            if st.button("Disorientation", key="disorientation_btn"):
                record_validation_status("disorientation")
                st.rerun()
        with col_btns4:
            if st.button("Deviation", key="deviation_btn"):
                record_validation_status('deviation')
                st.rerun()
        with col_btns5:
            if st.button("N/A", key="skip_btn"):
                record_validation_status(None)
                st.rerun()
    else:
        st.info("Please answer previous question(s) before proceeding.")


else:
    if st.session_state.stopped_early:
        st.warning("⚠️ Validation stopped early by user.")
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
        st.session_state.unit_results = []
        st.session_state.ocr_results = []
        st.session_state.stopped_early = False
        st.rerun()

####################################################################################################