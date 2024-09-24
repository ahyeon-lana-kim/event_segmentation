import re
import os
from typing import List
from transformers import GPT2Tokenizer

def read_file(file_path: str) -> str:
    """Read the content of a file and return it as a string."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_into_scenes(content: str) -> List[str]:
    """Split the content into scenes based on blank lines."""
    scenes = re.split(r'\n\s*\n', content)
    return [scene.strip() for scene in scenes if scene.strip()]

def create_tokenized_chunks(scenes: List[str], chunk_size: int = 1000, overlap: int = 5) -> List[str]:
    """Create overlapping chunks from the list of scenes using tokenization."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    chunks = []
    current_chunk = []
    current_size = 0

    for scene in scenes:
        scene_tokens = tokenizer.encode(scene)
        scene_size = len(scene_tokens)

        if current_size + scene_size > chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            overlap_scenes = current_chunk[-overlap:]
            current_chunk = overlap_scenes
            current_size = sum(len(tokenizer.encode(s)) for s in overlap_scenes)

        current_chunk.append(scene)
        current_size += scene_size

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks

def save_chunks(chunks: List[str], output_dir: str) -> None:
    """Save chunks to individual files in the specified directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, chunk in enumerate(chunks):
        output_file = os.path.join(output_dir, f'chunk_{i+1}.txt')
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(chunk)

def main(file_path: str, chunk_size: int = 1000, overlap: int = 5) -> None:
    """Main function to process the file and create tokenized chunks."""
    content = read_file(file_path)
    scenes = split_into_scenes(content)
    chunks = create_tokenized_chunks(scenes, chunk_size, overlap)

    output_dir = 'chunked_output'
    save_chunks(chunks, output_dir)

    print(f"Total chunks created: {len(chunks)}")
    print(f"Chunks saved in directory: {output_dir}")

if __name__ == "__main__":
    file_path = './'
    main(file_path, chunk_size=1000, overlap=5)
