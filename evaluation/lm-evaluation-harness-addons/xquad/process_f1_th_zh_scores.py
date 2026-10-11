#!/usr/bin/env python3
"""
This script processes JSONL files from an input folder and extracts F1 scores.
For files containing 'th' or 'zh', it recalculates F1 scores using language-specific tokenization.
For other files, it copies the existing F1 scores.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import jieba
from pythainlp.tokenize import word_tokenize
import numpy as np


def tokenize_text(text: str, lang: str) -> List[str]:
    """
    Tokenize text based on language.
    
    Args:
        text: Input text to tokenize
        lang: Language code ('th' for Thai, 'zh' for Chinese)
    
    Returns:
        List of tokens
    """
    if lang == 'th':
        # Thai tokenization using pythainlp
        tokens = word_tokenize(text, engine="newmm")
        return tokens
    elif lang == 'zh':
        # Chinese tokenization using jieba
        tokens = list(jieba.cut(text))
        return tokens
    else:
        # Default: split by whitespace
        return text.split()


def compute_f1_score(target: str, response: str, lang: str) -> float:
    """
    Compute F1 score between target and response using tokenization.
    
    Args:
        target: Target/reference answer
        response: Model's response
        lang: Language code
    
    Returns:
        F1 score (float between 0 and 1)
    """
    if not target or not response:
        return 0.0
    
    # Tokenize both target and response
    target_tokens = set(tokenize_text(str(target).strip(), lang))
    response_tokens = set(tokenize_text(str(response).strip(), lang))
    
    if len(target_tokens) == 0 and len(response_tokens) == 0:
        return 1.0
    if len(target_tokens) == 0 or len(response_tokens) == 0:
        return 0.0
    
    # Compute precision, recall, and F1
    common_tokens = target_tokens & response_tokens
    tp = len(common_tokens)
    fp = len(response_tokens) - tp
    fn = len(target_tokens) - tp
    
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 4)


def process_file(input_path: Path, output_path: Optional[Path] = None, lang: str = None) -> List[float]:
    """
    Process a single JSONL file.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file (optional)
        lang: Language code (detected from filename if None)
    
    Returns:
        List of F1 scores from the file
    """
    # Detect language from filename if not provided
    if lang is None:
        if 'th' in input_path.name:
            lang = 'th'
        elif 'zh' in input_path.name:
            lang = 'zh'
        else:
            lang = 'other'
    
    print(f"Processing {input_path.name} (lang: {lang})...")
    
    results = []
    f1_scores = []
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line)
                
                if lang in ['th', 'zh']:
                    # Recalculate F1 score for Thai and Chinese
                    target = data.get('target', '')
                    filtered_resps = data.get('filtered_resps', [''])[0]
                    
                    f1_score = compute_f1_score(target, filtered_resps, lang)
                    
                    # Create output entry with F1 score
                    output_entry = {
                        'doc_id': data.get('doc_id'),
                        'f1': f1_score
                    }
                else:
                    # Copy existing F1 score for other languages
                    f1_score = data.get('f1', 0.0)
                    output_entry = {
                        'doc_id': data.get('doc_id'),
                        'f1': f1_score
                    }
                
                f1_scores.append(output_entry['f1'])
                results.append(output_entry)
                
            except json.JSONDecodeError as e:
                print(f"  Warning: Error parsing line {line_num}: {e}")
                continue
    
    # Write output only when an output path is provided
    if output_path is not None:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for result in results:
                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"  Saved {len(results)} entries to {output_path.name}")
    else:
        print(f"  Processed {len(results)} entries (no output file written)")
    
    return f1_scores


def main():
    parser = argparse.ArgumentParser(
        description='Process JSONL files to extract/recalculate F1 scores'
    )
    parser.add_argument(
        'input_folder',
        type=str,
        help='Input folder containing JSONL files'
    )
    parser.add_argument(
        'output_folder',
        type=str,
        nargs='?',
        default=None,
        help='Output folder for F1 score files'
    )
    
    args = parser.parse_args()
    
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder) if args.output_folder else None
    
    # Validate input folder
    if not input_folder.exists():
        print(f"Error: Input folder '{input_folder}' does not exist")
        return
    
    if not input_folder.is_dir():
        print(f"Error: Input path '{input_folder}' is not a directory")
        return
    
    # Create output folder only when provided
    if output_folder is not None:
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Output folder: {output_folder}")
    else:
        print("Running in stats-only mode")
    
    # Find all JSONL files with _xquad_ pattern.
    # If the top-level folder has no matches, automatically look inside the
    # single child subfolder, which is how these runs are organized.
    search_root = input_folder
    jsonl_files = list(search_root.glob('*_xquad_*.jsonl'))
    if not jsonl_files:
        subdirs = [path for path in input_folder.iterdir() if path.is_dir()]
        if len(subdirs) == 1:
            search_root = subdirs[0]
            jsonl_files = list(search_root.glob('*_xquad_*.jsonl'))
            print(f"No matches at top level; searching subfolder: {search_root}")
    
    if not jsonl_files:
        print(f"Warning: No JSONL files with '_xquad_' pattern found in {input_folder}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL file(s) with '_xquad_' pattern")
    print()
    
    # Track F1 scores by language
    f1_by_lang = {}
    
    # Process each file
    for input_file in sorted(jsonl_files):
        output_file = output_folder / input_file.name if output_folder is not None else None
        
        # Determine language
        if 'th' in input_file.name:
            lang = 'th'
        elif 'zh' in input_file.name:
            lang = 'zh'
        else:
            # Extract language code from filename (e.g., _xquad_en_, _xquad_fr_, etc.)
            # Pattern: *_xquad_XX_* where XX is language code
            parts = input_file.name.split('_xquad_')
            if len(parts) > 1:
                lang_part = parts[1].split('_')[0]
                lang = lang_part if lang_part else 'other'
            else:
                lang = 'other'
        
        f1_scores = process_file(input_file, output_file, lang)
        
        if lang not in f1_by_lang:
            f1_by_lang[lang] = []
        f1_by_lang[lang].extend(f1_scores)
    
    print()
    print("="*60)
    print("F1 Score Statistics by Language")
    print("="*60)

    # Show only Thai and Chinese in the final summary
    for lang in ['th', 'zh']:
        if lang not in f1_by_lang:
            continue
        scores = np.array(f1_by_lang[lang])
        mean = np.mean(scores)
        std = np.std(scores)
        print(f"Language: {lang:8s} | Mean: {mean:.4f} | Std: {std:.4f} | Count: {len(scores)}")
    
    print("="*60)
    print(f"All files processed successfully!")


if __name__ == '__main__':
    main()
