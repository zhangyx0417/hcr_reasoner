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


def process_intention_query(story: str, query: str, answer: str) -> Tuple[str, int, int]:
    """
    Process intention queries using vanilla prompting.
    
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
    
    return pred, correct, incorrect


class CausalReasoner:
    """Main class for handling causal reasoning tasks"""
    
    def __init__(self, data_path: str, step1_path: str, step2_path: str, results_path: str):
        self.examples = read_json(data_path)
        self.prompt_1 = read_txt(step1_path)
        self.prompt_2 = read_txt(step2_path)
        self.results_path = results_path
        
        # Initialize counters
        self.start_idx, self.corr, self.inco, self.cause_idx, self.c_corr, self.c_inco = resume(results_path)
        
    def process_example(self, idx: int, example: Dict[str, Any]) -> None:
        """Process a single example and save results"""
        
        if idx < self.start_idx:
            return
            
        story, query, answer = example['story'], example['question'], example['answer']
        
        if 'intentionally' in query:
            result = self._handle_intention_query(story, query, answer)
        else:
            result = self._handle_causation_query(story, query, answer)
            
        self._save_result(idx, result)
        self._print_progress(idx)

    def _handle_intention_query(self, story: str, query: str, answer: str) -> Dict[str, Any]:
        """Handle intention-related queries using vanilla prompting"""
        pred, correct, incorrect = process_intention_query(
            story, query, answer
        )
        
        # Update counters
        if pred == answer:
            self.corr += 1
        else:
            self.inco += 1
            
        return {
            'story': story,
            'query': query,
            'pred': pred,
            'gold': answer,
            'correct': pred == answer
        }

    def _handle_causation_query(self, story: str, query: str, answer: str) -> Dict[str, Any]:
        """Handle causation-related queries using multi-stage reasoning"""
        self.cause_idx += 1
        
        # Stage 1: Get causal setting
        causal_setting = self._get_causal_setting(story, query)

        # Stage 2: Get causal factors
        causal_factors = self._get_causal_factors(story, query, causal_setting)
        
        # Stage 3: Make prediction
        pred, response = self._make_prediction(story, query, causal_setting, causal_factors)
        
        # Update counters
        if pred == answer:
            self.corr += 1
            self.c_corr += 1
        else:
            self.inco += 1
            self.c_inco += 1
            
        return {
            'story': story,
            'query': query,
            'causal_setting': causal_setting,
            'causal_factors': causal_factors,
            'response': response,
            'pred': pred,
            'gold': answer,
            'correct': pred == answer
        }

    def _get_causal_setting(self, story: str, query: str) -> Dict[str, Any]:
        """Get causal setting through Stage 1 reasoning"""
        messages = [
            {'role': 'system', 'content': 'You are an expert in the field of actual causality and causal judgment. Given the story and query of a logic-based causal judgment problem, you can effectively assist in solving the problem following the instructions provided.'}, 
            {'role': 'user', 'content': self.prompt_1.format(story=story, query=query)}
        ]
        
        raw_causal_setting = chat_completions(messages)
        return parse_json_response(raw_causal_setting)

    def _get_causal_factors(self, story: str, query: str, causal_setting: Dict[str, Any]) -> Dict[str, Any]:
        """Get causal factors through Stage 2 reasoning"""
        messages = [
            {'role': 'system', 'content': 'You are an expert in the field of actual causality and causal judgment. Given the story and query of a logic-based causal judgment problem, you can effectively assist in solving the problem following the instructions provided.'}, 
            {'role': 'user', 'content': self.prompt_1.format(story=story, query=query)},
            {'role': 'assistant', 'content': json.dumps(causal_setting)},
            {'role': 'user', 'content': self.prompt_2.replace('\{', '{').replace('\}', '}')}
        ]
        
        raw_causal_factors = chat_completions(messages)
        return parse_json_response(raw_causal_factors)

    def _make_prediction(self, story: str, query: str, causal_setting: Dict[str, Any], causal_factors: Dict[str, Any]) -> Tuple[str, str]:
        """Make final prediction based on causal analysis"""
        messages = [
            {'role': 'system', 'content': 'You are an expert in the field of actual causality and causal judgment. Given the story and query of a logic-based causal judgment problem, you can effectively assist in solving the problem following the instructions provided.'}, 
            {'role': 'user', 'content': self.prompt_1.format(story=story, query=query)},
            {'role': 'assistant', 'content': json.dumps(causal_setting)},
            {'role': 'user', 'content': self.prompt_2.replace('\{', '{').replace('\}', '}')},
            {'role': 'assistant', 'content': json.dumps(causal_factors)},
            {'role': 'user', 'content': 'Based on the causal setting and the values of causal factors, answer the query.\nAnswer (Yes or No?): '}
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
        
        return pred, raw_answer

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
    STEP_1_PATH = str()  # Path to prompt of Stage 1
    STEP_2_PATH = str()  # Path to prompt of Stage 2
    RESULTS_PATH = str()  # Path to saved results
    
    # Create reasoner and process examples
    reasoner = CausalReasoner(DATA_PATH, STEP_1_PATH, STEP_2_PATH, RESULTS_PATH)
    for idx, example in enumerate(reasoner.examples):
        reasoner.process_example(idx, example)


if __name__ == '__main__':
    main()
