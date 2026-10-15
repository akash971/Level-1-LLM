"""
=============================================================================
Basic Next-Word Predictor (Bigram Language Model in Pure Python)
=============================================================================
This script demonstrates the foundational concept of how Language Models (LLMs)
predict the next word:
1. Tokenize text into individual words.
2. Build a transition dictionary: current_word -> {next_word: count}.
3. Pick the next word based on frequency (probabilities).
4. Chain predictions together to form a sentence.
=============================================================================
"""

import re
import os
from collections import defaultdict


def load_and_tokenize(file_path: str) -> list[str]:
    """
    Reads a text file and converts it into a list of clean, lowercase words.
    
    Why tokenization matters:
    Computers don't see words natively—they see raw text strings. Tokenization
    breaks the text into discrete chunks ("tokens" or words) so we can analyze
    patterns and transitions between them.
    """
    # Check if the file exists before trying to open it
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Step 1: Normalize to lowercase so "The" and "the" are treated as the same word.
    clean_text = raw_text.lower()

    # Step 2: Extract words using regular expressions:
    # '\b[a-z0-9]+\b' finds sequences of alphanumeric characters bounded by word boundaries,
    # stripping away punctuation like periods, commas, quotation marks, etc.
    tokens = re.findall(r"\b[a-z0-9]+\b", clean_text)
    
    return tokens


def build_next_word_model(tokens: list[str]) -> dict[str, dict[str, int]]:
    """
    Constructs a transition frequency dictionary (a Bigram model).
    
    Data Structure:
    {
        "the": {"quick": 3, "lazy": 2, "warm": 2},
        "quick": {"brown": 2, "rabbit": 1},
        ...
    }
    
    Each key is a 'current_word', and its value is a dictionary mapping each
    possible 'next_word' to the number of times it appeared immediately after.
    """
    # A defaultdict(lambda: defaultdict(int)) lets us safely increment counts
    # without having to write multiple `if key not in dict` checks.
    transitions = defaultdict(lambda: defaultdict(int))

    # Loop through all adjacent pairs (tokens[i], tokens[i+1])
    for i in range(len(tokens) - 1):
        current_word = tokens[i]
        next_word = tokens[i + 1]

        # Increment how many times `next_word` followed `current_word`
        transitions[current_word][next_word] += 1

    # Convert back to standard dicts for clean printing and pure Python structure
    return {word: dict(next_words) for word, next_words in transitions.items()}


def predict_next_word(current_word: str, model: dict[str, dict[str, int]]) -> str | None:
    """
    Given a current word, finds the most frequently occurring next word.
    (This is known as 'Greedy Search' in modern LLMs).
    """
    current_word = current_word.lower()

    # If the word is not in our dictionary, we cannot predict what comes next
    if current_word not in model:
        return None

    next_candidates = model[current_word]
    if not next_candidates:
        return None

    # Find the candidate word that has the maximum frequency count.
    # In case of ties, max() consistently picks the first one encountered.
    best_next_word = max(next_candidates, key=next_candidates.get)
    return best_next_word


def generate_sentence(start_word: str, model: dict[str, dict[str, int]], length: int = 5) -> str:
    """
    Generates a sentence of a given length (default: 5 words) starting with `start_word`.
    
    Steps:
    1. Begin with the user's start word.
    2. Predict the next word based on highest frequency.
    3. Update current word to the predicted word.
    4. Repeat until reaching `length` words or hitting a dead end (a word with no follow-up).
    """
    start_word = start_word.strip().lower()

    # Verify that the starting word exists in our vocabulary
    if start_word not in model:
        available_sample = list(model.keys())[:10]
        return (
            f"Error: The word '{start_word}' was not found in the vocabulary.\n"
            f"Here are some words you can try: {', '.join(available_sample)}..."
        )

    # Initialize the sentence list with the starting word
    sentence_words = [start_word]
    current = start_word

    # Generate the remaining words (up to total target length)
    for _ in range(length - 1):
        next_word = predict_next_word(current, model)

        # Handle dead ends (e.g., if the word only occurred at the very end of the file)
        if next_word is None:
            break

        sentence_words.append(next_word)
        current = next_word  # Move forward in the chain

    # Format the generated words into a neat, capitalized sentence with a period
    formatted_sentence = " ".join(sentence_words).capitalize() + "."
    return formatted_sentence


def main():
    # Path to our training corpus
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "sample.txt")

    print("=" * 60)
    print("  PURE PYTHON NEXT-WORD PREDICTION & SENTENCE GENERATOR")
    print("=" * 60)
    print(f"Loading and analyzing text from: {file_path}")

    # Step 1: Tokenize
    tokens = load_and_tokenize(file_path)
    print(f"Total tokens processed: {len(tokens)}")

    # Step 2: Build the Bigram frequency dictionary
    model = build_next_word_model(tokens)
    print(f"Unique vocabulary words: {len(model)}")
    print("-" * 60)

    # Show a small preview of how the dictionary looks internally
    preview_words = ["the", "quick", "brown"]
    print("Sample transition frequencies stored in dictionary:")
    for w in preview_words:
        if w in model:
            print(f"  '{w}' -> {model[w]}")
    print("-" * 60)

    # Display helpful suggestions for the user
    sample_starts = ["the", "quick", "brown", "a", "sleepy"]
    print(f"Suggested starting words to try: {', '.join(sample_starts)}")
    print("Type 'exit' or 'quit' to stop.\n")

    # Step 3: Interactive Loop
    while True:
        try:
            user_input = input("Enter a starting word: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Exiting. Goodbye!")
            break

        # Generate a 5-word sentence
        result = generate_sentence(user_input, model, length=5)
        print(f"Generated 5-word sentence:\n  -> {result}\n")


if __name__ == "__main__":
    main()
