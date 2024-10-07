import numpy as np
import argparse
import pickle
import os
import re

def load_text_from_file(file_path):
    """Load text from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def create_binary_vector_from_text(text, is_gpt=False):
    """Create a binary vector from the text where each element represents a line or an event."""
    if is_gpt:
        # For GPT output, split by "Event X:" pattern
        events = re.split(r'Event \d+:', text)[1:]  # [1:] to skip the empty first element
        vector = np.ones(len(events), dtype=int)
    else:
        # For human annotations, split by lines
        lines = text.split('\n')
        vector = np.zeros(len(lines), dtype=int)
        vector[0] = 1  # First line is always the start of a paragraph
        
        for i in range(1, len(lines)):
            if not lines[i].strip():
                for j in range(i+1, len(lines)):
                    if lines[j].strip():
                        vector[j] = 1
                        break
    
    return vector

def print_vector_with_text(text, vector, is_gpt=False):
    """Print the binary vector alongside the original text for easy comparison."""
    if is_gpt:
        events = re.split(r'Event \d+:', text)[1:]  # [1:] to skip the empty first element
        for i, (event, value) in enumerate(zip(events, vector), 1):
            print(f"Event {i} - {value}: {event.strip()[:100]}...")  # Print first 100 chars of each event
    else:
        lines = text.split('\n')
        for i, (line, value) in enumerate(zip(lines, vector)):
            print(f"{value}: {line}")

def save_vector_as_pickle(vector, output_path):
    """Save the binary vector as a pickle file."""
    with open(output_path, 'wb') as f:
        pickle.dump(vector, f)
    print(f"Binary vector saved to {output_path}")

def main(file_path, output_path, is_gpt):
    """Main function to load text, create a binary vector, display results, and save the vector."""
    text = load_text_from_file(file_path)
    binary_vector = create_binary_vector_from_text(text, is_gpt)
    
    print("Binary Vector:")
    print(binary_vector)
    print("\nVector with Text:")
    print_vector_with_text(text, binary_vector, is_gpt)
    
    if is_gpt:
        print(f"\nTotal events: {len(binary_vector)}")
    else:
        print(f"\nTotal lines: {len(binary_vector)}")
    print(f"Number of segments: {np.sum(binary_vector)}")
    
    save_vector_as_pickle(binary_vector, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a binary vector from text annotations and save as pickle.")
    parser.add_argument("file_path", help="Path to the file containing text annotations")
    parser.add_argument("--output", help="Path to save the output pickle file")
    parser.add_argument("--gpt", action="store_true", help="Flag to indicate if the input is GPT-3 output")
    args = parser.parse_args()

    if args.output is None:
        # If no output path is provided, use the input filename with .pkl extension
        args.output = os.path.splitext(args.file_path)[0] + '.pkl'
    
    main(args.file_path, args.output, args.gpt)
