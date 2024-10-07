import numpy as np
from scipy.spatial.distance import hamming
import pickle
import json
import argparse

def load_binary_vector(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def manual_agreement(vec1, vec2):
    return np.mean(vec1 == vec2)

def get_p_valueShuffle(bounds1, bounds2, n_rand=1000):
    distance_real = hamming(bounds1, bounds2)
    rand_dists = np.ones(n_rand,) * np.inf
    for rr in range(n_rand):
        vec2 = bounds2.copy()
        np.random.shuffle(vec2)
        rand_dists[rr] = hamming(bounds1, vec2)
    p_val = np.sum(rand_dists <= distance_real)/n_rand
    return p_val, rand_dists

def get_p_valueCircShuffle(bounds1, bounds2, n_rand=1000):
    distance_real = hamming(bounds1, bounds2)
    rand_dists = np.ones(n_rand,) * np.inf
    for rr in range(n_rand):
        vec2 = bounds2.copy()
        vec2 = np.roll(vec2, np.random.randint(0,len(vec2)))
        rand_dists[rr] = hamming(bounds1, vec2)
    p_val = np.sum(rand_dists <= distance_real)/n_rand
    return p_val, rand_dists

def compare_annotations(gpt_vector, human_vector):
    if len(gpt_vector) != len(human_vector):
        raise ValueError("GPT and human vectors have different lengths.")
    
    true_distance = hamming(gpt_vector, human_vector)
    p_value_shuffle, rand_dists_shuffle = get_p_valueShuffle(gpt_vector, human_vector)
    p_value_circ, rand_dists_circ = get_p_valueCircShuffle(gpt_vector, human_vector)
    
    return true_distance, p_value_shuffle, p_value_circ, np.std(rand_dists_shuffle), np.std(rand_dists_circ)


def main(gpt_file, human_file, output_file):
    gpt_vector = load_binary_vector(gpt_file)
    human_vector = load_binary_vector(human_file)

    print(f"GPT vector length: {len(gpt_vector)}")
    print(f"Human vector length: {len(human_vector)}")
    print(f"GPT events: {np.sum(gpt_vector)}")
    print(f"Human events: {np.sum(human_vector)}")
    
    print("First 20 elements of GPT vector:", gpt_vector[:20])
    print("First 20 elements of Human vector:", human_vector[:20])

    min_length = min(len(gpt_vector), len(human_vector))
    gpt_vector = gpt_vector[:min_length]
    human_vector = human_vector[:min_length]

    true_distance, p_value_shuffle, p_value_circ, std_distance_shuffle, std_distance_circ = compare_annotations(gpt_vector, human_vector)
    
    print("\nResults:")
    print("=" * 40)
    print(f"True Hamming distance: {true_distance:.4f}")
    print("\nP-values:")
    print(f"  Random Shuffle:   {p_value_shuffle:.4f}")
    print(f"  Circular Shuffle: {p_value_circ:.4f}")
    print("\nStandard deviations of Hamming distances:")
    print(f"  Random Shuffle:   {std_distance_shuffle:.4f}")
    print(f"  Circular Shuffle: {std_distance_circ:.4f}")
    print("=" * 40)

    results = {
        "true_hamming_distance": float(true_distance),
        "p_value_random_shuffle": float(p_value_shuffle),
        "p_value_circular_shuffle": float(p_value_circ),
        "standard_deviation_random_shuffle": float(std_distance_shuffle),
        "standard_deviation_circular_shuffle": float(std_distance_circ)
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {output_file}")

# [The rest of the script remains the same]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare GPT and human annotation binary vectors.")
    parser.add_argument("gpt_file", help="Path to the pickle file containing GPT binary vector")
    parser.add_argument("human_file", help="Path to the pickle file containing human binary vector")
    parser.add_argument("--output", default="comparison_results.json", help="Path to save the output JSON file (default: comparison_results.json)")
    args = parser.parse_args()

    main(args.gpt_file, args.human_file, args.output)
