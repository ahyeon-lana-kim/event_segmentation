import os
import logging
from openai import OpenAI
import pickle
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt', quiet=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def process_chunk(chunk, prompt, max_retries=3, delay=1):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI that segments stories into events as instructed."},
                    {"role": "user", "content": prompt + chunk}
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error processing chunk (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))  # Exponential backoff
            else:
                return None

def main(chunk_dir, story_name, version=""):
    prompt = (
        "An event is an ongoing coherent situation. The following story needs to be copied and segmented into"
        "events. Copy the following story word-for-word and start a new line whenever one"
        "event ends and another begins. This is the story: "
    )
  
prompt_suffix = "\n This is a word-for-word copy of the same story that is segmented into " + version +  "events: "
  
    chunk_files = sorted(
        [f for f in os.listdir(chunk_dir) if f.startswith('chunk_') and f.endswith('.txt')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )

    all_events = []

    for chunk_file in chunk_files:
        logging.info(f"Processing {chunk_file}")
        with open(os.path.join(chunk_dir, chunk_file), 'r', encoding='utf-8') as fp:
            chunk_text = fp.read()
        
        logging.info(f"Chunk text length: {len(chunk_text)} characters")

        response = process_chunk(chunk_text, prompt)
        
        if response:
            events = response.split('\n\n')  # Split on blank lines
            events = [event.replace('\n', ' ').strip() for event in events if event.strip()]
            
            if events:
                all_events.extend(events)
                logging.info(f"Extracted {len(events)} events from {chunk_file}")
                for i, event in enumerate(events, 1):
                    logging.info(f"Event {i} length: {len(event)} characters, {len(sent_tokenize(event))} sentences")
            else:
                logging.warning(f"No events extracted from {chunk_file} after processing")
        else:
            logging.warning(f"Failed to process {chunk_file}")

    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{story_name}_version_{version.strip() if version else "standard"}_Events.pkl'

    with open(output_file, 'wb') as f:
        pickle.dump(all_events, f)

    logging.info(f"Processing complete. Total events extracted: {len(all_events)}")
    logging.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    dir = ''
    story_name = ''
    main(, story_name)
  
