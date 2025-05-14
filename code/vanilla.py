from openai import OpenAI
import json
import os
from utils import read_json, read_txt, resume, partial_upper, parse_json_response
from typing import List, Dict, Any, Tuple, Optional


client = ...


def chat_completions(messages: List[Dict[str, str]], model: str = 'gpt-4-0613', 
                     temperature: float = 0) -> str:
    """
    Implement your chat_completions() for different models here.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: Model identifier string
        temperature: Sampling temperature
        
    Returns:
        Response content as string
    """
    raise NotImplementedError("Please implement chat_completions() for your specific model API")


def process_queries(story: str, query: str, answer: str) -> Tuple[str, int, int, str]:
    """
    Process queries using vanilla prompting.
    
    Args:
        story: The story text
        query: The query about intention
        answer: The correct answer
        
    Returns:
        Tuple containing prediction, correctness counters, and other metadata
    """
    messages = [
        {'role': 'system', 'content': 'You are an expert in the field of actual causality and causal judgment. Given the story and query of a logic-based causal judgment problem, you can effectively solve it.'}, 
        {'role': 'user', 'content': f'Story: {story}\nQuery: {query}\n\nAnswer (Yes or No?): '}
    ]
    
    raw_answer = chat_completions(messages)
    
    # Determine prediction
    if 'Yes' in raw_answer:
        pred = 'Yes'
    elif 'No' in raw_answer:
        pred = 'No'
    else:
        # Force a false prediction if unclear
        pred = 'Yes' if answer == 'No' else 'No'
    
    # Track correctness
    correct = 1 if pred == answer else 0
    incorrect = 0 if pred == answer else 1
    
    return pred, correct, incorrect, raw_answer


class CausalReasoner:
    """Main class for handling causal reasoning tasks"""
    
    def __init__(self, data_path: str, results_path: str):
        self.examples = read_json(data_path)
        self.results_path = results_path
        
        # Initialize counters
        self.start_idx, self.corr, self.inco, self.cause_idx, self.c_corr, self.c_inco = resume(results_path)
        
    def process_example(self, idx: int, example: Dict[str, Any]) -> None:
        """Process a single example and save results"""
        
        if idx < self.start_idx:
            return
            
        story, query, answer = example['story'], example['question'], example['answer']
        result = self._handle_queries(story, query, answer)
            
        self._save_result(idx, result)
        self._print_progress(idx)

    def _handle_queries(self, story: str, query: str, answer: str) -> Dict[str, Any]:
        """Handle intention-related queries using vanilla prompting"""
        pred, correct, incorrect, explanations = process_queries(
            story, query, answer
        )

        flag = 'intentionally' in query
        if not flag:
            self.cause_idx += 1
        
        # Update counters
        if pred == answer:
            self.corr += 1
            if not flag:
                self.c_corr += 1
        else:
            self.inco += 1
            if not flag:
                self.c_inco += 1
            
        return {
            'story': story,
            'query': query,
            'response': explanations,
            'pred': pred,
            'gold': answer,
            'correct': pred == answer
        }

    def _save_result(self, idx: int, result: Dict[str, Any]) -> None:
        """Save processing result to file"""
        with open(self.results_path, 'a') as f:
            f.write(json.dumps(result) + '\n')

    def _print_progress(self, idx: int) -> None:
        """Print progress information"""
        print(f'[{idx + 1}/{len(self.examples)}]  '
              f'Acc: {round(self.corr / (idx + 1), 4)}  '
              f'Inco: {self.inco}  '
              f'Cau: {round(self.c_corr / self.cause_idx, 4)}  '
              f'Inco: {self.c_inco}')


def main():
    """Main execution function"""
    DATA_PATH = str()  # Path to AC-Bench or BBH-CJ
    RESULTS_PATH = str()  # Path to saved results
    
    # Create reasoner and process examples
    reasoner = CausalReasoner(DATA_PATH, RESULTS_PATH)
    for idx, example in enumerate(reasoner.examples):
        reasoner.process_example(idx, example)


if __name__ == '__main__':
    main()
