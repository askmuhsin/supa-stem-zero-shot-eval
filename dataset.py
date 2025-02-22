from typing import List
from datasets import load_dataset as hf_load_dataset
import pandas as pd
import logging


def load_dataset(
    dataset_name: str = "Supa-AI/STEM-en-ms",
    language: str = "en",
    split: str = "eval",
    filter_figures: bool = True
) -> pd.DataFrame:
    """    
    Args:
        dataset_name: Name of the dataset in Hugging Face
        language: Language code ('en' or 'ms')
        split: Dataset split to load
        filter_figures: If True, only return questions without figures
    """
    logging.info(f"Loading dataset {dataset_name} with language '{language}' and split '{split}'")
    dataset = hf_load_dataset(
        dataset_name,
        name=f"data_{language}",
        split=split
    )
    
    data = pd.DataFrame(dataset)
    logging.info(f"Dataset loaded. Shape: {data.shape}")
    
    if filter_figures:
        logging.info("Filtering out questions with figures")
        data = data[data["Figures"].apply(lambda x: isinstance(x, list) and len(x) == 0)]
        logging.info(f"Filtered dataset shape: {data.shape}")
    
    return data


def format_question_prompt(item: dict) -> tuple[str, str]:
    options = item['Options'] if isinstance(item['Options'], list) else eval(item['Options'])
    formatted_options = '\n'.join(f"    {opt}" for opt in options)
    prompt = (
        f"Instruction: Solve the problem, ensure your final answer includes the choice letter (A, B, C, or D).\n"
        f"Question: {item['Questions']}\n"
        f"Choices:\n"
        f"{formatted_options}\n\n"
        f"Solution:  "
    )
    return prompt, item['Answers']


def get_formatted_questions(data: pd.DataFrame, sample_size: int = None) -> List[tuple[int, str, str]]:
    if sample_size:
        data = data.sample(n=sample_size)
    return [(idx, *format_question_prompt(row.to_dict())) for idx, row in data.iterrows()]


if __name__ == "__main__":
    data = load_dataset()
    questions = get_formatted_questions(data, sample_size=2)
    print("\n---\n".join([f"Index: {idx}\n{prompt}\n{answer}" for idx, prompt, answer in questions]))
    logging.info("Example questions printed.")
