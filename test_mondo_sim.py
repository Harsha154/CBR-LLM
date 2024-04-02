import subprocess

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def levenshtein_similarity(input_str, comparison_str):
    lev_distance = levenshtein_distance(input_str, comparison_str)
    max_distance = max(len(input_str), len(comparison_str))
    similarity = 1 - lev_distance / max_distance
    return similarity

def run_bash_script(search_term):
    bash_script = "./find_disease_words.sh"
    result = subprocess.run([bash_script, search_term], capture_output=True, text=True)
    print(result.stdout)

    filename_safe_search_term = search_term.replace(" ", "_")
    output_file_path = f"/home/hs875/Llama-2/cases/test_results/{filename_safe_search_term}_results.txt"
    return output_file_path

def main():
    # Example strings for distance and similarity calculation
    s1 = "example"
    s2 = "samples"
    
    distance = levenshtein_distance(s1, s2)
    similarity_score = levenshtein_similarity(s1, s2)
    print(f"Levenshtein Distance between {s1} and {s2} = {distance}")
    print(f"Similarity score between {s1} and {s2} = {similarity_score}")

    # Example search term for bash script
    search_term = "diabetes"
    output_file_path = run_bash_script(search_term)
    print(f"Results saved in: {output_file_path}")

if __name__ == "__main__":
    main()
