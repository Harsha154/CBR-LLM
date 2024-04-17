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


# List to store each row's data
data_to_export = []
for number, desc in random_rows_dict.items():
    LLM_with_CBR = cbr_llm.run_llm_with_cbr(number, desc, cases_data, wcl = 0.35, wct = 0.65)
    LLM_without_CBR = llm_no_cbr.run_llm_with_no_cbr(number, desc)
    

    data_to_export.append({
        "ind_text_id": LLM_without_CBR["ind_text_id"],  # or LLM_with_CBR["ind_text"] as they should be the same
        "ind_text": LLM_without_CBR["ind_text"],
        "LLM_NO_CBR Prompt": LLM_without_CBR["prompt"],
        "LLM_NO_CBR Score": LLM_without_CBR["LLM_NO_CBR Score: "],
        "LLM_Output" : LLM_without_CBR["LLM Output"],
        "CBR_LLM Prompt": LLM_with_CBR["prompt"],
        "CBR_LLM Score": LLM_with_CBR["CBR_LLM Score: "],
        "Instructions (Prompt Item 3)": LLM_with_CBR["Instructions (Prompt Item 3)"],
        "CBR+LLM Output" : LLM_with_CBR["CBR+LLM Output"] ,
        "Case ID": LLM_with_CBR["Case ID"]

    })

df = pd.DataFrame(data_to_export)

# Save the DataFrame to a CSV file
csv_file_path = 'experiment_results.csv'
df.to_csv(csv_file_path, index=False)

print(f"CSV file created at {csv_file_path}")

