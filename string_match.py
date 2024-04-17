def count_matches(input_string, file_path = "mondo_names_only.txt"):
    # Load the database
    with open(file_path, 'r') as file:
        database = {line.strip().lower() for line in file}
    matches_count = "Fail"
    # Split the input string into words to check for all possible substrings
    matches = set()
    words = input_string.lower().split()
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            substring = " ".join(words[i:j])
            if substring in database:
                matches_count = "Pass"
                matches.add(substring)
    if not matches:
        return matches_count # true if match is found, false is no matches
    else:
        return matches

