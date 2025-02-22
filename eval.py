import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from dataset import load_dataset, get_formatted_questions
from model import get_completion
from answer_evaluator import get_final_selection


@dataclass
class EvalResult:
    total_questions: int
    correct_answers: int
    accuracy: float
    duration_seconds: float
    timestamp: str
    mistakes: List[Dict]
    trace: Dict


class ModelEvaluator:
    def __init__(self, cache_file: str = "eval_cache.json"):
        self.logger = logging.getLogger(__name__)
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
        self.mistakes = []
        self.trace = {}

    def _load_cache(self) -> Dict:
        if self.cache_file.exists():
            self.logger.info(f"Loading cache from {self.cache_file}")
            with open(self.cache_file) as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        self.logger.info(f"Saving cache to {self.cache_file}")
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def evaluate_dataset(self, sample_size: Optional[int] = None, force_rerun: List[int] = None) -> EvalResult:
        force_rerun = force_rerun or []
        
        self.logger.info("Loading dataset...")
        data = load_dataset()
        problems = get_formatted_questions(data, sample_size)
        total = len(problems)
        
        self.logger.info(f"Evaluating {total} questions...")
        correct = 0
        self.mistakes = []
        self.trace = {}

        start_time = datetime.now()
        
        try:
            for idx, (id_, question, correct_answer) in enumerate(problems, 1):
                if str(id_) in self.cache and id_ not in force_rerun:
                    self.logger.debug(f"Using cached result for question {id_}")
                    cached = self.cache[str(id_)]
                    self.trace[id_] = cached
                    if cached['model_answer'] == cached['correct_answer']:
                        correct += 1
                    else:
                        self.mistakes.append(cached)
                    continue

                self.logger.info(f"Processing question {idx}/{total} (ID: {id_})")

                model_response = get_completion(question)
                model_answer, reasoning = get_final_selection(model_response)
                
                result = {
                    'id': id_,
                    'question': question,
                    'correct_answer': correct_answer,
                    'model_answer': model_answer,
                    'model_response': model_response,
                    'reasoning': reasoning
                }
                
                self.cache[str(id_)] = result
                self.trace[id_] = result
                
                if model_answer == correct_answer:
                    correct += 1
                else:
                    self.mistakes.append(result)
                
                if idx % 10 == 0:
                    self._save_cache()
                
                accuracy = (correct / idx) * 100
                self.logger.info(f"Current accuracy: {accuracy:.2f}% ({correct}/{idx})")
                
        except KeyboardInterrupt:
            self.logger.warning("Evaluation interrupted by user")
        finally:
            self._save_cache()
        
        duration = datetime.now() - start_time
        final_accuracy = (correct / total) * 100
        
        result = EvalResult(
            total_questions=total,
            correct_answers=correct,
            accuracy=final_accuracy,
            duration_seconds=duration.total_seconds(),
            timestamp=datetime.now().isoformat(),
            mistakes=self.mistakes,
            trace=self.trace
        )
        
        self._log_summary(result)
        return result

    def _log_summary(self, result: EvalResult):
        self.logger.info("\nEvaluation completed:")
        self.logger.info(f"Accuracy: {result.accuracy:.2f}% ({result.correct_answers}/{result.total_questions})")
        self.logger.info(f"Duration: {timedelta(seconds=int(result.duration_seconds))}")
        self.logger.info(f"Number of mistakes: {len(result.mistakes)}")
        
        if result.mistakes:
            self.logger.info("\nMistake summary:")
            for mistake in result.mistakes[:5]:  # Show first 5 mistakes
                self.logger.info(f"ID {mistake['id']}: Expected {mistake['correct_answer']}, Got {mistake['model_answer']}")
            if len(result.mistakes) > 5:
                self.logger.info(f"... and {len(result.mistakes) - 5} more mistakes")


def save_results(results: EvalResult, output_file: str):
    with open(output_file, 'w') as f:
        json.dump(asdict(results), f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Evaluate model performance on STEM dataset')
    parser.add_argument('--sample-size', type=int, help='Number of questions to evaluate')
    parser.add_argument('--cache-file', default='eval_cache.json', help='Cache file location')
    parser.add_argument('--output-file', default='eval_results.json', help='Output file for results')
    parser.add_argument('--force-rerun', type=int, nargs='*', help='Question IDs to re-evaluate')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    evaluator = ModelEvaluator(cache_file=args.cache_file)
    results = evaluator.evaluate_dataset(
        sample_size=args.sample_size,
        force_rerun=args.force_rerun
    )
    
    save_results(results, args.output_file)
    logging.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
