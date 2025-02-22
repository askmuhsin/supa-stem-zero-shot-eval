import re
import logging
from model import get_completion, ModelConfig


ANSWER_EVALUATOR_CONFIG = ModelConfig(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    max_tokens=1000,
    base_url="https://router.huggingface.co/hf-inference/v1"
)


def extract_answer_from_completion(completion: str) -> str:
    answer_match = re.search(r'<answer>\s*([ABCDX])\s*</answer>', completion)
    return answer_match.group(1) if answer_match else 'X'


def get_final_selection(student_answer_sheet: str) -> str:
    prompt = f'''You are an automated grading assistant tasked with identifying the final answer selection from a student's answer sheet. Your goal is to extract only the student's selected answer choice, without evaluating its correctness.
<student_answer_sheet>
{student_answer_sheet}
</student_answer_sheet>

Process:
1. Identify all mentioned choices and eliminated options
2. Find definitive selection language or uncertainty
3. Return XML with analysis and final letter

Output format:
<answer_sheet_breakdown>
- Choices mentioned: [letters]
- Eliminated: [letters]
- Selection evidence: [text excerpt]
- Uncertainty: [yes/no + reason if yes]
</answer_sheet_breakdown>
<answer>[Single letter: A, B, C, D, or X]</answer>'''

    completion = None
    try:
        completion = get_completion(prompt, config=ANSWER_EVALUATOR_CONFIG)
        return extract_answer_from_completion(completion), completion
    except Exception as e:
        logging.error(f"Error during completion: {e}")
        return 'X', completion


if __name__ == "__main__":
    test_answer = """I think the answer is B because it matches the pattern. 
    I considered A but it doesn't work because of the constraints.
    Definitely not C or D."""
    
    result = get_final_selection(test_answer)
    print(f"Extracted answer: {result}")
