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


def evaluate_causal_responsibility(messages: List[Dict[str, str]], focal_event: str, 
                                 event_list: List[str], responsibility_factors: str) -> Tuple[str, str]:
    """
    Evaluate causal responsibility based on specified factors.
    
    Args:
        messages: Current conversation messages
        focal_event: The event being evaluated
        event_list: List of relevant events
        responsibility_factors: Description of factors determining responsibility
        
    Returns:
        Tuple of prediction and explanation
    """
    prompt = (f'Define responsibility as the relative degree (more, less, or equally) to which a causal '
             f'event causally contributes to the outcome event, relative to other causal events specified. '
             f'Here, assume responsibility is only determined by {responsibility_factors}.\n\n'
             f'Return Yes if based on the story, the focal causal event "{focal_event}" is equally or more '
             f'responsible relative to other causal events in the list {event_list}, else No. '
             f'Then, explain briefly based on the story.')
    
    messages.append({'role': 'user', 'content': prompt})
    raw_pred = chat_completions(messages)
    
    if any(x in raw_pred.lower() for x in ['yes']):
        return 'Yes', raw_pred
    elif any(x in raw_pred.lower() for x in ['no']):
        return 'No', raw_pred
    else:
        raise ValueError('Invalid prediction format')


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
        pred, explanations, events = self._make_prediction(causal_setting, causal_factors)
        
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
            'explanations': explanations,
            'events': events,
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

    def _handle_sufficient_cause(self, focal_event: str, causal_setting: Dict[str, Any],
                               causal_factors: Dict[str, Any], order: int,
                               norm_violated: bool, behavior_intended: bool) -> Tuple[str, str]:
        """Handle sufficient but not necessary causes"""
        # Find all sufficient causes
        sufficient_causes = [
            evt for evt, factors in causal_factors.items()
            if factors['sufficient'] and not factors['necessary']
        ]
        
        # Compare temporal orders
        all_orders = [causal_setting['causal_events'][evt]['order'] for evt in sufficient_causes]
        
        if len(set(all_orders)) != 1:  # Different temporal orders
            if order == min(all_orders):
                return 'Yes', (
                    f'"{focal_event}" is a cause of "{list(causal_setting["outcome_event"].keys())[0]}", '
                    f'since "{focal_event}" occurs the earliest among multiple disjunctive causal events.'
                )
            return 'No', (
                f'"{focal_event}" is not a cause of "{list(causal_setting["outcome_event"].keys())[0]}", '
                f'since "{focal_event}" does not occur the earliest among multiple disjunctive causal events.'
            )
        
        # If same temporal order, evaluate responsibility
        messages = [
            {'role': 'system', 'content': 'You are an expert in causal reasoning.'},
            {'role': 'user', 'content': (
                f'Define responsibility as the relative degree (more, less, or equally) to which a causal '
                f'event causally contributes to the outcome event, relative to other causal events specified. '
                f'Here, assume responsibility is only determined by normality (`norm_violated`) and '
                f'intention (`behavior_intended`).\n\nReturn Yes if based on the story, the focal causal '
                f'event "{focal_event}" is equally or more responsible relative to other causal events in '
                f'the list {sufficient_causes}, else No. Then, explain briefly based on the story.'
            )}
        ]
        
        pred, explanation = evaluate_causal_responsibility(
            messages, focal_event, sufficient_causes, 
            "normality (`norm_violated`) and intention (`behavior_intended`)"
        )
        return pred, explanation

    def _handle_halpern_pearl_cause(self, focal_event: str, causal_setting: Dict[str, Any],
                                   causal_factors: Dict[str, Any], order: int,
                                   norm_violated: bool, behavior_intended: bool) -> Tuple[str, str]:
        """Handle Halpern-Pearl causes that are not sufficient"""
        if norm_violated:
            return 'Yes', (
                f'"{focal_event}" is a cause of "{list(causal_setting["outcome_event"].keys())[0]}", '
                f'since "{focal_event}" is an actual cause, and it violates a norm.'
            )
        
        if behavior_intended:
            return 'Yes', (
                f'"{focal_event}" is a cause of "{list(causal_setting["outcome_event"].keys())[0]}", '
                f'since "{focal_event}" is an actual cause, and it is an intended behavior of an agent.'
            )
        
        # Find all Halpern-Pearl causes
        hp_causes = [
            evt for evt, factors in causal_factors.items()
            if not factors['sufficient'] and factors['halpern_pearl']
        ]
        
        # Compare temporal orders
        all_orders = [causal_setting['causal_events'][evt]['order'] for evt in hp_causes]
        
        if len(set(all_orders)) != 1:
            messages = [
                {'role': 'system', 'content': 'You are an expert in causal reasoning.'},
                {'role': 'user', 'content': (
                    f'Define responsibility as the relative degree (more, less, or equally) to which a causal '
                    f'event causally contributes to the outcome event, relative to other causal events specified. '
                    f'Here, assume responsibility is only determined by temporal order (`order`).\n\nReturn Yes '
                    f'if based on the story, the focal causal event "{focal_event}" is more responsible relative '
                    f'to other causal events in the list {hp_causes}, else No. Then, explain briefly based on '
                    f'the story.'
                )}
            ]
            
            pred, explanation = evaluate_causal_responsibility(
                messages, focal_event, hp_causes, "temporal order (`order`)"
            )
            return pred, explanation
        
        return 'No', (
            f'"{focal_event}" is not a cause of "{list(causal_setting["outcome_event"].keys())[0]}", '
            f'since "{focal_event}" is an actual cause, but it neither violates a norm nor is an '
            f'intended behavior of an agent.'
        )

    def _make_prediction(self, causal_setting: Dict[str, Any], 
                        causal_factors: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
        """Make final prediction based on causal analysis"""
        preds = []
        explanations = []
        S_list = []  # Sufficient causes list
        H_list = []  # Halpern-Pearl causes list
        
        causal_events = causal_setting['causal_events']
        outcome_event = causal_setting['outcome_event']
        
        for focal_event, factors in causal_factors.items():
            if not causal_events[focal_event]['focal']:
                continue
                
            # Extract factors from Stage 1
            occur = causal_events[focal_event]['occur']
            order = causal_events[focal_event]['order']
            
            # Extract factors from Stage 2
            sufficient = factors['sufficient']
            necessary = factors['necessary']
            halpern_pearl = factors['halpern_pearl']
            norm_violated = factors['norm_violated']
            behavior_intended = factors['behavior_intended']
            
            # Make prediction based on causal reasoning rules
            if sufficient and necessary:
                preds.append('Yes')
                explanations.append(
                    f'"{focal_event}" is a cause of "{list(outcome_event.keys())[0]}", '
                    f'since "{focal_event}" is both sufficient and necessary.'
                )
            elif sufficient and not necessary:
                pred, explanation = self._handle_sufficient_cause(
                    focal_event, causal_setting, causal_factors, order, 
                    norm_violated, behavior_intended
                )
                preds.append(pred)
                explanations.append(explanation)
                S_list.append(focal_event)
            elif not sufficient and halpern_pearl:
                pred, explanation = self._handle_halpern_pearl_cause(
                    focal_event, causal_setting, causal_factors, order,
                    norm_violated, behavior_intended
                )
                preds.append(pred)
                explanations.append(explanation)
                H_list.append(focal_event)
            else:
                preds.append('No')
                explanations.append(
                    f'"{focal_event}" is not a cause of "{list(outcome_event.keys())[0]}", '
                    f'since "{focal_event}" is neither sufficient nor necessary.'
                )
        
        if not preds:  # No focal event found
            return 'No', ['No focal event identified'], []
            
        final_pred = 'Yes' if 'Yes' in preds else 'No'
        events = S_list if S_list else H_list
        
        return final_pred, explanations, events

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
