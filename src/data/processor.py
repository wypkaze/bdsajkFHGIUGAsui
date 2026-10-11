"""
Data processing utilities including BM25 retrieval and prompt construction.
"""

from typing import Callable
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from ..utils.text_utils import get_first_k_tokens
from ..utils.extractors import (
    extract_option,
    extract_citation_title,
    extract_movie,
    extract_news_headline,
    extract_product_review,
    extract_scholarly_title,
    extract_tweet_paraphrasing,
)


# Task to extractor mapping
TASK_EXTRACTORS = {
    "citation": extract_citation_title,
    "movie_tagging": extract_movie,
    "news_headline": extract_news_headline,
    "product_rating": extract_product_review,
    "scholarly_title": extract_scholarly_title,
    "tweet_paraphrase": extract_tweet_paraphrasing,
}


class DataProcessor:
    """Process training and test data with BM25 retrieval."""
    
    def __init__(
        self,
        task_name: str,
        prompt_template: dict,
        k: int = 1,
        add_profile: bool = False,
    ):
        """
        Initialize data processor.
        
        Args:
            task_name: Name of the task
            prompt_template: Prompt template dictionary
            k: Number of history items to retrieve (0=disabled)
            add_profile: Whether to add user profile to prompt
        """
        self.task_name = task_name
        self.prompt_template = prompt_template
        self.k = k
        self.add_profile = add_profile
        self.extractor = TASK_EXTRACTORS.get(task_name, extract_citation_title)
    
    def build_prompt(self, query: dict, user_profile: str = "") -> tuple[str, str]:
        """
        Build prompt and full_prompt for a query.
        
        Args:
            query: Query dictionary with 'input' and 'gold'
            user_profile: User profile text (optional)
        
        Returns:
            Tuple of (prompt, full_prompt)
        """
        if self.task_name == "citation":
            question = query['input']
            article = extract_citation_title(question)
            option1 = extract_option(question, 1)
            option2 = extract_option(question, 2)
            prompt = self.prompt_template[self.task_name]['prompt'].format(
                article, option1, option2
            )
            full_prompt = self.prompt_template[self.task_name]['full_prompt'].format(
                article, option1, option2, query['gold']
            )
        else:
            article = get_first_k_tokens(self.extractor(query['input']), 768)
            prompt = self.prompt_template[self.task_name]['prompt'].format(article)
            full_prompt = self.prompt_template[self.task_name]['full_prompt'].format(
                article, query['gold']
            )
        
        return prompt, full_prompt
    
    def add_history(
        self,
        prompt: str,
        full_prompt: str,
        user: dict,
        article: str,
    ) -> tuple[str, str]:
        """
        Add retrieved history to prompts using BM25.
        
        Args:
            prompt: Base prompt
            full_prompt: Base full_prompt
            user: User data with 'profile'
            article: Article text for retrieval query
        
        Returns:
            Tuple of (prompt_with_history, full_prompt_with_history)
        """
        if self.k <= 0:
            return prompt, full_prompt
        
        # Prepare history
        visible_history_list = user['profile']
        for p in visible_history_list:
            for key, value in p.items():
                p[key] = get_first_k_tokens(value, 368)
        
        # Build history list
        history_list = [
            self.prompt_template[self.task_name]['retrieval_history'].format(**p)
            for p in visible_history_list
        ]
        
        # BM25 retrieval
        tokenized_corpus = [doc.split(" ") for doc in history_list]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = self.prompt_template[self.task_name]["retrieval_query_wokey"].format(
            article
        ).split(' ')
        retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=self.k)
        
        # Add history to prompts
        history_string = "".join(retrieved_history)
        prompt = f"### User History:\n{history_string}\n\n" + prompt
        full_prompt = f"### User History:\n{history_string}\n\n" + full_prompt
        
        return prompt, full_prompt
    
    def add_user_profile(
        self,
        prompt: str,
        full_prompt: str,
        user_profile: str,
    ) -> tuple[str, str]:
        """
        Add user profile to prompts.
        
        Args:
            prompt: Base prompt
            full_prompt: Base full_prompt
            user_profile: User profile text
        
        Returns:
            Tuple of (prompt_with_profile, full_prompt_with_profile)
        """
        if not self.add_profile or not user_profile:
            return prompt, full_prompt
        
        prompt = f"### User Profile:\n{user_profile}\n\n" + prompt
        full_prompt = f"### User Profile:\n{user_profile}\n\n" + full_prompt
        
        return prompt, full_prompt
    
    def process_user(self, user: dict, all_profile: dict) -> list[dict]:
        """
        Process all queries for a user.
        
        Args:
            user: User data with 'user_id', 'profile', 'query'
            all_profile: Dictionary of user profiles
        
        Returns:
            List of processed data points
        """
        user_id = str(user['user_id'])
        user_profile = all_profile.get(user_id, "") if self.add_profile else ""
        
        results = []
        for q in user['query']:
            # Build base prompts
            prompt, full_prompt = self.build_prompt(q, user_profile)
            
            # Get article for retrieval
            if self.task_name == "citation":
                article = extract_citation_title(q['input'])
            else:
                article = get_first_k_tokens(self.extractor(q['input']), 768)
            
            # Add history
            prompt, full_prompt = self.add_history(prompt, full_prompt, user, article)
            
            # Add profile
            prompt, full_prompt = self.add_user_profile(prompt, full_prompt, user_profile)
            
            results.append({"prompt": prompt, "full_prompt": full_prompt})
        
        return results


def process_train_data(
    train_data: list,
    prompt_template: dict,
    all_profile: dict,
    task_name: str,
    k: int = 1,
    add_profile: bool = False,
) -> list[dict]:
    """
    Process all training data.
    
    Args:
        train_data: List of user training data
        prompt_template: Prompt template dictionary
        all_profile: Dictionary of user profiles
        task_name: Name of the task
        k: Number of history items to retrieve
        add_profile: Whether to add user profile
    
    Returns:
        List of processed training samples
    """
    processor = DataProcessor(task_name, prompt_template, k, add_profile)
    
    processed_data = []
    for user in tqdm(train_data, desc="Processing training data"):
        results = processor.process_user(user, all_profile)
        processed_data.extend(results)
    
    return processed_data


def process_test_data(
    test_data: list,
    prompt_template: dict,
    all_profile: dict,
    task_name: str,
    k: int = 1,
    add_profile: bool = False,
) -> list[tuple[str, str]]:
    """
    Process all test data.
    
    Args:
        test_data: List of user test data
        prompt_template: Prompt template dictionary
        all_profile: Dictionary of user profiles
        task_name: Name of the task
        k: Number of history items to retrieve
        add_profile: Whether to add user profile
    
    Returns:
        List of (question_id, prompt) tuples
    """
    processor = DataProcessor(task_name, prompt_template, k, add_profile)
    
    test_prompts = []
    for user in test_data:
        user_id = str(user['user_id'])
        user_profile = all_profile.get(user_id, "") if add_profile else ""
        
        # Prepare BM25 if needed
        bm25 = None
        history_list = []
        if k > 0:
            visible_history_list = user['profile']
            for p in visible_history_list:
                for key, value in p.items():
                    p[key] = get_first_k_tokens(value, 368)
            
            history_list = [
                prompt_template[task_name]['retrieval_history'].format(**p)
                for p in visible_history_list
            ]
            tokenized_corpus = [doc.split(" ") for doc in history_list]
            bm25 = BM25Okapi(tokenized_corpus)
        
        for q in user['query']:
            # Build base prompt
            if task_name == 'citation':
                test_question = q['input']
                test_article = extract_citation_title(test_question)
                option1, option2 = extract_option(test_question, 1), extract_option(test_question, 2)
                test_prompt = prompt_template[task_name]['prompt'].format(test_article, option1, option2)
            else:
                test_question = q['input']
                test_article = processor.extractor(test_question)
                test_prompt = prompt_template[task_name]['prompt'].format(test_article)
            
            # Add history
            if k > 0 and bm25:
                tokenized_query = prompt_template[task_name]['retrieval_query_wokey'].format(
                    test_article
                ).split(" ")
                retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=k)
                history_string = "".join(retrieved_history)
                test_prompt = f"### User History:\n{history_string}\n\n" + test_prompt
            
            # Add profile
            if add_profile and user_profile:
                test_prompt = f"### User Profile:\n{user_profile}\n\n" + test_prompt
            
            test_prompts.append((q['id'], test_prompt))
    
    return test_prompts
