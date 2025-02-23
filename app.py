import streamlit as st
import json
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="STEM Model Evaluation",
    page_icon="🔬",
    layout="wide"
)

def load_cache(cache_file: str = "eval_cache.json") -> dict:
    with open(cache_file) as f:
        return json.load(f)

def create_dataframe(cache: dict) -> pd.DataFrame:
    records = []
    for _, item in cache.items():
        records.append({
            'id': item['id'],
            'question': item['question'].split('\n')[1],  # Get just the question part
            'correct_answer': item['correct_answer'],
            'model_answer': item['model_answer'],
            'is_correct': item['correct_answer'] == item['model_answer'],
            'full_question': item['question'],
            'model_response': item['model_response'],
            'reasoning': item['reasoning']
        })
    return pd.DataFrame(records)

def main():
    st.title("🔬 Supa-AI/STEM-en-ms Evaluation Analysis")
    st.subheader("Model : Deepseek-v3 | temperature 0.1 | top_p 0.1")
    st.info("""
        **Evaluation Methodology:**
        - Zero-shot evaluation using a two-stage approach
        - Main model (DeepSeek-V3) solves STEM problems with detailed reasoning
        - Mistral-7B-Instruct-v0.2 extracts final answer choice from solution
        - Each question is prompted with: 'Instruction: Solve the problem, ensure your final answer includes the choice letter (A, B, C, or D).'
    """)
    
    cache = load_cache()
    df = create_dataframe(cache)
    
    # Summary Metrics
    st.header("📊 Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Questions", 
            len(df),
            help="Total number of questions evaluated"
        )
    
    accuracy = (df['is_correct'].sum() / len(df)) * 100
    with col2:
        st.metric(
            "Accuracy", 
            f"{accuracy:.1f}%",
            help="Percentage of correct answers"
        )
    
    with col3:
        st.metric(
            "Incorrect Answers",
            (len(df) - df['is_correct'].sum()),
            help="Number of questions answered incorrectly"
        )
    
    # Question Analysis
    st.header("🔍 Question Analysis")
    
    # Filters
    col1, col2 = st.columns([1, 2])
    with col1:
        filter_type = st.selectbox(
            "Filter questions",
            ["All Questions", "Correct Only", "Incorrect Only"]
        )
    
    with col2:
        search_term = st.text_input(
            "Search in questions",
            placeholder="Type to search..."
        )
    
    # Apply filters
    if filter_type == "Correct Only":
        df_filtered = df[df['is_correct']]
    elif filter_type == "Incorrect Only":
        df_filtered = df[~df['is_correct']]
    else:
        df_filtered = df
        
    if search_term:
        df_filtered = df_filtered[
            df_filtered['question'].str.contains(search_term, case=False)
        ]
    
    # Display questions
    st.subheader(f"Showing {len(df_filtered)} questions")
    
    for _, row in df_filtered.iterrows():
        with st.expander(
            f"Question {row['id']} - "
            f"{'✅ Correct' if row['is_correct'] else '❌ Incorrect'} - "
            f"{row['question'][:100]}..."
        ):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Question (Model Prompt)**")
                st.markdown(row['full_question'])
                st.markdown("**Correct Answer:** " + row['correct_answer'])
                st.markdown("**Model Answer:** " + row['model_answer'])
            
            with col2:
                st.markdown("**Model's Reasoning**")
                st.markdown(row['model_response'])
                st.markdown("**Answer Extraction**")
                st.markdown(row['reasoning'])

if __name__ == "__main__":
    main()
