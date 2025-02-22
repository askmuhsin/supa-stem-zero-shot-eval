# STEM Dataset Zero-Shot Evaluation

A zero-shot evaluation pipeline for the [STEM-en-ms dataset](https://huggingface.co/datasets/Supa-AI/STEM-en-ms) using a two-stage approach:
1. Main model solves the STEM problem
2. Lightweight model extracts the final answer

## Quick Start

```bash
# Clone the repo
git clone [your-repo-url]
cd stem-dataset-eval

# Set up environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Set up API tokens in .env
echo "HUGGING_FACE_API_TOKEN=your_token_here" > .env

# Run evaluation
python eval.py --sample-size 10  # Test on 10 questions
python eval.py  # Run full evaluation
```

## Project Structure

```
├── dataset.py         # Dataset loading and formatting
├── model.py          # Model configuration and inference
├── answer_evaluator.py # Answer extraction from model output
├── eval.py           # Main evaluation pipeline
└── notebooks/        # Development notebooks
```


## Usage Examples

### As a Module

```python
from eval import ModelEvaluator

evaluator = ModelEvaluator(cache_file="my_cache.json")
results = evaluator.evaluate_dataset(sample_size=10)

print(f"Accuracy: {results.accuracy:.2f}%")
print(f"Mistakes: {len(results.mistakes)}")
```

### Command Line

```bash
# Basic usage
python eval.py

# Evaluation options
python eval.py --sample-size 10 --cache-file custom_cache.json
python eval.py --force-rerun 123 456  # Re-run specific questions
python eval.py --debug  # Enable debug logging
```

## Implementation Details

The evaluation uses a two-stage approach:

1. **Problem Solving**: Uses a large language model to solve STEM problems zero shot.
2. **Answer Extraction**: A lightweight model extracts the final answer (A/B/C/D) from the solution. 

I did this to understand if the model is trying to solve the problem without worrying about the styling, or outputing a choice immediately, without having to think, can it improve the results.

## Environment Variables

Required in `.env` file:
```
HUGGING_FACE_API_TOKEN=your_token_here
```

## Cache Files

- `eval_cache.json`: Stores model responses and evaluations
- `eval_results.json`: Contains evaluation metrics and analysis

## Development

Check the notebooks for development history and experiments:
- `01_dataset.ipynb`: Initial dataset exploration
- `02_model_calibrate.ipynb`: Model calibration
- `03_run_benchmark_1.ipynb`: First benchmark run
- `04_dev_benchmark_2.ipynb`: Pipeline development
- `05_run_evaluator.ipynb`: Final evaluation runs
