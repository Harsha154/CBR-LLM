import pandas as pd
import random 
import cbr_llm
import llm_no_cbr
# Import prompt text from the organized_indications.csv file
csv_file_path = "organized_indications.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)
random_rows = df.sample(n=10)
# Create a dictionary with 'Number' of the entry as keys and 'Description' as values
random_rows_dict = pd.Series(random_rows['Description'].values, index=random_rows['Number']).to_dict()

# location of case base csv file
cases_csv = "/home/hs875/Llama-2/cases/updated_case_base_3-31.csv"

cases_data = pd.read_csv(cases_csv)

#prompt_id for bash script to save string matches
prompt_id = 0 
prompt_id_2 = 100

# List to store each row's data
data_to_export = []
for number, desc in random_rows_dict.items():
    prompt_id+=1
    prompt_id_2+=1
    LLM_with_CBR = cbr_llm.run_llm_with_cbr(number, desc, cases_data, prompt_id, wcl = 0.35, wct = 0.65)
    LLM_without_CBR = llm_no_cbr.run_llm_with_no_cbr(number, desc, prompt_id_2)
    # find the max similar score, the result with the least edit distance
    llm_normalized_scores = [value['Normalized Similarity Score'] for key, value in LLM_without_CBR.items() if isinstance(value, dict) and 'Normalized Similarity Score' in value]
    llm_cbr_normalized_scores = [value['Normalized Similarity Score'] for key, value in LLM_with_CBR.items() if isinstance(value, dict) and 'Normalized Similarity Score' in value]

    # Find the max normalized score
    llm_max_normalized_score = max(llm_normalized_scores) if llm_normalized_scores else 0
    llm_cbr_max_normalized_score = max(llm_cbr_normalized_scores) if llm_cbr_normalized_scores else 0


    data_to_export.append({
        "ind_text": LLM_without_CBR["ind_text"],  # or LLM_with_CBR["ind_text"] as they should be the same
        "LLM_NO_CBR Score": llm_max_normalized_score,
        "CBR_LLM Score": llm_cbr_normalized_scores,
        "Instructions (Prompt Item 3)": LLM_with_CBR["Instructions (Prompt Item 3)"],
        "Case ID": LLM_with_CBR["Case ID"]
    })

df = pd.DataFrame(data_to_export)

# Save the DataFrame to a CSV file
csv_file_path = 'experiment_results.csv'
df.to_csv(csv_file_path, index=False)

print(f"CSV file created at {csv_file_path}")
    
