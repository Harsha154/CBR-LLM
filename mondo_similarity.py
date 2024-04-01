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

def run_bash_script(search_term, prompt_id):
    bash_script = "./find_disease_words.sh"
    # Ensure both arguments are strings
    search_term = str(search_term)
    prompt_id_str = str(prompt_id)
    result = subprocess.run([bash_script, search_term, prompt_id_str], capture_output=True, text=True)
    print(f"debug: {result}")
    output_file_path = f"/home/hs875/Llama-2/cases/test_results/{prompt_id_str}_results.txt"
    return output_file_path

def get_similary_score_with_mondo(search_term, prompt_id):
    results_file = run_bash_script(search_term, prompt_id)

    # Process the results file
    with open(results_file, 'r') as file:
        lines = file.readlines()
        # Assume each line in the file is a search result to be compared
        output_dict = {}
        for line in lines:
            distance = levenshtein_distance(search_term, line.strip())
            similarity_score = levenshtein_similarity(search_term, line.strip())
            output_dict[line] = {"Levenshtein Distance": distance,
                                        "Normalized Similarity Score": similarity_score}

    return output_dict

if __name__ == "__main__":
    output = get_similary_score_with_mondo("Dental", 19)
    print(output)
