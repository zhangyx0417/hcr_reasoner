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


def process_queries(story: str, query: str, answer: str) -> Tuple[str, int, int, List[str]]:
    """
    Process queries using manual CoT prompting.
    
    Args:
        story: The story text
        query: The query about intention
        answer: The correct answer
        
    Returns:
        Tuple containing prediction, correctness counters, and other metadata
    """
    messages = [
        {'role': 'system', 'content': 'You are an expert in the field of actual causality and causal judgment. Given the story and query of a logic-based causal judgment problem, you can effectively solve it.'}, 
        {'role': 'user', 'content': f'Here we will provide three chain-of-thought exemplars, followed by a binary question that needs to be answered.\n\nStory: Frank T., had an ongoing dispute with his neighbor over a stretch of land and one day decided to shoot his neighbor in the body. Frank T. had no experience with guns, his hand slipped on the barrel of the gun, and the shot went wild. Nonetheless, the bullet bounced off a large boulder several feet away and hit the neighbor\'s body, causing significant injury.\nQuery: Did Frank T. intentionally shoot his neighbor in the body?\nOptions:\n- Yes\n- No\nAnswer (with chain of thought): No. Here in this question, we are told that "Frank T. had no experience with guns, his hand slipped on the barrel of the gun, and the shot went wild." A typical person would assume that this passage suggests that Frank T. had no intention of shooting and injuring someone and that the bullet accidentally hit the neighbor\'s body; therefore, we conclude that Frank T. did not intentionally hit his neighbor. So the answer is No.\n\nStory: Suzy and Billy are working on a project that is very important for our nation\'s security. The boss tells them both: "Be sure that you are here at exactly 9 am. It is absolutely essential that you arrive at that time." Both Billy and Suzy arrive at 9 am. As it happens, there was a motion detector installed in the room where they arrived. The motion detector was set up to be triggered if at least one person appeared in the room at the same time. So the motion detector went off.\nQuery: Did Billy cause the motion detector to go off?\nOptions:\n- Yes\n- No\nAnswer (with chain of thought): Yes. Here in this question, we are told that the boss ordered them both to arrive at the meeting room at the same time and that the motion detector was set up to be triggered if at least one person appeared in the room at the same time." A typical person would assume that the person probably meant to say the detector was set up to be triggered if "both persons" appeared in the room at the same time, not at least one person, since otherwise the phrase "at the same time" would not make much sense in that sentence. Because the motion detector went off, a typical person would therefore come to the conclusion that both Suzy and Billy triggered the motion detector to go off; hence, Billy did indeed cause the motion detector to go off. So the answer is Yes.\n\nStory: George and his sister Lena reunite at their parents\' house for Thanksgiving. Whereas George just got into medical school, Lena is unhappy in her marriage and recently lost her job. Over the course of the day, George and Lena get into a number of heated arguments. Later in the afternoon they play a game of darts. They split the first two games, and the third game is close until the end. Who will win comes down to George\'s last shot. If he hits a high point region, he wins; if he hits a low point region, Lena wins. George thinks of the difficult time Lena is having, and he really wants to let her win. He aims the dart at the low point region. He sets up his shot and the dart lands in the low point region. After his shot, Lena wins the game and is very happy.\nQuery: Did George hit the low point region intentionally?\nOptions:\n- Yes\n- No\nAnswer (with chain of thought): Yes. Here in this question, we are told that "He aims the dart at the low point region." A typical person might therefore think George did intentionally hit the low point region, because he wanted to lift up the spirit of his sister Lena. So the answer is Yes.\n\nStory: {story}\nQuery: {query}\n\nAnswer (Yes or No?): '}
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
