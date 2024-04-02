#!/bin/bash

# Accept search term and prompt_id as arguments
search_term=$1
prompt_id=$2

# Define the directory for saving results
results_dir="/home/hs875/Llama-2/cases/test_results"

# Check if the results directory exists, if not, create it
mkdir -p "$results_dir"

# Perform the fuzzy search and save the results to a file
cat "mondo_names_only.txt" | fzf -f "$search_term" > "${results_dir}/${prompt_id}_results.txt"

echo "Results saved to ${results_dir}/${prompt_id}_results.txt"

