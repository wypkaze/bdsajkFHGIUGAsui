#!/usr/bin/env python3
"""
Clean output files by removing repeated prompt content based on prompt.json templates.
"""

import json
import re
import argparse
from pathlib import Path


def load_prompt_templates(prompt_path: str) -> dict:
    """Load prompt templates from JSON file."""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_real_output(output_text: str, task_name: str, prompt_templates: dict) -> str:
    """
    Extract the real output from the text by removing repeated prompt template content.
    
    Args:
        output_text: Raw model output text
        task_name: Task name
        prompt_templates: Dictionary of prompt templates
    
    Returns:
        Cleaned output text
    """
    if task_name not in prompt_templates:
        return output_text.strip()
    
    templates = prompt_templates[task_name]
    cleaned = output_text
    
    # Template patterns to remove
    patterns_to_remove = []
    
    # 1. Remove User Profile section if present
    if '### User Profile:' in cleaned:
        # Find all ### User Profile: positions, keep content after the last one
        profile_splits = cleaned.split('### User Profile:')
        if len(profile_splits) > 1:
            # The last split contains the actual content
            cleaned = profile_splits[-1]
            # Handle User History if still present
            if '### User History:' in cleaned:
                history_splits = cleaned.split('### User History:')
                if len(history_splits) > 1:
                    cleaned = history_splits[-1]
    
    # 2. Remove User History section if still present
    if '### User History:' in cleaned:
        history_splits = cleaned.split('### User History:')
        if len(history_splits) > 1:
            cleaned = history_splits[-1]
    
    # 3. Remove task-specific instruction prompts
    if task_name == 'product_rating':
        # Remove product_rating instruction
        instruction_pattern = r"### User Instruction:\s*What is the score of the following review on a scale of 1 to 5\? just answer with 1, 2, 3, 4, or 5 without further explanation\.\s*"
        cleaned = re.sub(instruction_pattern, '', cleaned, flags=re.DOTALL)
        
        # Extract content after the last "Score:"
        if 'Score:' in cleaned:
            score_splits = cleaned.split('Score:')
            if len(score_splits) > 1:
                # Get content after the last Score:
                after_score = score_splits[-1].strip()
                # Return the digit if present
                if after_score and after_score[0].isdigit():
                    return after_score[0]
                return after_score
    
    elif task_name == 'movie_tagging':
        # Remove movie_tagging instruction
        instruction_pattern = r"### User Instruction:\s*Which tag does this movie relate to among the following tags\? Just answer with the tag name without further explanation\. tags: \[.*?\]\s*"
        cleaned = re.sub(instruction_pattern, '', cleaned, flags=re.DOTALL)
        
        # Extract content after the last "Tag:"
        if 'Tag:' in cleaned:
            tag_splits = cleaned.split('Tag:')
            if len(tag_splits) > 1:
                # Get content after the last Tag:
                after_tag = tag_splits[-1].strip()
                return after_tag
    
    elif task_name == 'news_headline':
        # Remove news_headline instruction
        instruction_pattern = r"### User Instruction:\s*Generate a headline for the following article\.\s*"
        cleaned = re.sub(instruction_pattern, '', cleaned, flags=re.DOTALL)
        
        # Extract content after the last "Headline:"
        if 'Headline:' in cleaned:
            headline_splits = cleaned.split('Headline:')
            if len(headline_splits) > 1:
                after_headline = headline_splits[-1].strip()
                return after_headline
    
    elif task_name == 'scholarly_title':
        # Remove scholarly_title instruction
        instruction_pattern = r"### User Instruction:\s*Generate a title for the following abstract of a paper\.\s*"
        cleaned = re.sub(instruction_pattern, '', cleaned, flags=re.DOTALL)
        
        # Extract content after the last "Title:"
        if 'Title:' in cleaned:
            title_splits = cleaned.split('Title:')
            if len(title_splits) > 1:
                after_title = title_splits[-1].strip()
                return after_title
    
    elif task_name == 'tweet_paraphrase':
        # Remove tweet_paraphrase instruction
        instruction_pattern = r"### User Instruction:\s*Paraphrase the following text into tweet without any explanation before or after it\.\s*"
        cleaned = re.sub(instruction_pattern, '', cleaned, flags=re.DOTALL)
        
        # Extract content after the last "Tweet:"
        if 'Tweet:' in cleaned:
            tweet_splits = cleaned.split('Tweet:')
            if len(tweet_splits) > 1:
                after_tweet = tweet_splits[-1].strip()
                return after_tweet
    
    elif task_name == 'citation':
        # Remove citation instruction
        instruction_pattern = r"### User Instruction:\s*Identify the most relevant reference for the listed publication by the researcher\. Select the reference paper that is most closely related to the researcher's work\. Please respond with only the number that corresponds to the reference\.\s*"
        cleaned = re.sub(instruction_pattern, '', cleaned, flags=re.DOTALL)
        
        # Extract content after the last "Answer:"
        if 'Answer:' in cleaned:
            answer_splits = cleaned.split('Answer:')
            if len(answer_splits) > 1:
                after_answer = answer_splits[-1].strip()
                return after_answer
    
    # General cleanup: remove remaining ### marker lines
    lines = cleaned.split('\n')
    result_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines starting with ###
        if stripped.startswith('###'):
            continue
        # Skip input markers like Description:, Review:, Article:, etc.
        if stripped.startswith('Description:') or stripped.startswith('Review:') or stripped.startswith('Article:') or stripped.startswith('Abstract:') or stripped.startswith('Text:') or stripped.startswith('Paper Title:'):
            continue
        result_lines.append(line)
    
    cleaned = '\n'.join(result_lines).strip()
    
    # If result is too long, it might contain the entire prompt, take first non-empty line
    if len(cleaned) > 500:
        lines = cleaned.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and len(line) < 300:
                return line
    
    return cleaned


def clean_output_file(input_path: str, output_path: str, prompt_templates: dict):
    """
    Clean a single output file.
    """
    # Infer task name from path
    task_name = None
    path_parts = input_path.split('/')
    for part in path_parts:
        # Match directory names like "product_rating-task_LoRA", exclude "outputs-task_LoRA"
        if '-task_LoRA' in part and part != 'outputs-task_LoRA':
            # Extract "product_rating" from "product_rating-task_LoRA"
            task_name = part.split('-task_LoRA')[0]
            break
    
    if not task_name:
        print(f"Warning: Could not infer task name from {input_path}")
        return
    
    print(f"Processing {input_path} (task: {task_name})")
    
    # Read file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Clean each output
    cleaned_count = 0
    for item in data.get('golds', []):
        original_output = item.get('output', '')
        
        # If output doesn't start with ###, it's already clean
        if not original_output.strip().startswith('###'):
            continue
        
        cleaned_output = extract_real_output(original_output, task_name, prompt_templates)
        
        if original_output != cleaned_output:
            cleaned_count += 1
            if cleaned_count <= 3:  # Only show first 3 examples
                print(f"  Cleaned [{item.get('id', 'unknown')}]")
                print(f"    Original: {original_output[:80]}...")
                print(f"    Cleaned:  {cleaned_output[:80]}...")
        
        item['output'] = cleaned_output
    
    # Save file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"✓ Cleaned {cleaned_count} items, saved to {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description='Clean output files by removing prompt templates')
    parser.add_argument('--input', '-i', type=str, default='outputs-task_LoRA', help='Input directory')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output directory (default: overwrite input)')
    parser.add_argument('--prompt', '-p', type=str, default='prompt/prompt.json', help='Path to prompt.json')
    
    args = parser.parse_args()
    
    # Load prompt templates
    prompt_templates = load_prompt_templates(args.prompt)
    print(f"Loaded prompt templates for: {list(prompt_templates.keys())}\n")
    
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir
    
    # Process all output files
    for json_file in input_dir.glob('**/*.json'):
        # Skip files in checkpoint directories
        if 'checkpoint' in str(json_file):
            continue
        
        # Only process output-*.json files
        if 'output' not in json_file.name:
            continue
        
        # Calculate output path
        if args.output:
            relative_path = json_file.relative_to(input_dir)
            out_file = output_dir / relative_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_file = json_file
        
        try:
            clean_output_file(str(json_file), str(out_file), prompt_templates)
        except Exception as e:
            print(f"✗ Error processing {json_file}: {e}\n")


if __name__ == '__main__':
    main()
