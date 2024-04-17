import torch
import pandas as pd
import re
import string_match
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize the device and model
torch.cuda.set_device(4)
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/Llama-2-7B-32K-Instruct")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/Llama-2-7B-32K-Instruct",
                                             trust_remote_code=True, torch_dtype=torch.float16).to(device)

def run_llm_with_cbr(ind_num, ind_text, cases_data, wcl = 0.35, wct = 0.65):
    input_ids = tokenizer.encode(ind_text, return_tensors="pt").to(device)
    with torch.no_grad():
        model_output = model(input_ids, output_hidden_states=True)
        input_embedding = model_output.hidden_states[-1].mean(dim=1)  # Average pooling over the sequence

    input_text_length = len(ind_text)
    # local cosine_sim
    local_weight_similarity_case_instructions = ""
    local_cosine_similarity_score = 0
    local_weighted_vector = 0
    # Convert the rows in cases.csv into dictionaries and compute local_cosine similarity
    for case_dict in cases_data.to_dict(orient='records'):
        case_text = case_dict["Text passage"]
        case_passage_len = case_dict["Passage length"]
        case_input_ids = tokenizer.encode(case_text, return_tensors="pt").to(device)

        with torch.no_grad():
            case_output = model(case_input_ids, output_hidden_states=True)
            case_embedding = case_output.hidden_states[-1].mean(dim=1)  # Average pooling over the sequence

        # Compute local_cosine similarity between the input_embedding and case_embedding
        local_cosine_sim_tensor = F.cosine_similarity(input_embedding, case_embedding, dim=1)
        local_cosine_sim = local_cosine_sim_tensor.item()

        # Compute local_length similarity between the input_text_length and case_passage_len
        local_length_sim = abs(case_passage_len-input_text_length)/max(input_text_length,case_passage_len)

        # Calculate weighted vector using weight parameters
        weighted_vector = (local_length_sim * wcl) + (local_cosine_sim * wct)

        if weighted_vector > local_weighted_vector:
            local_weighted_vector = weighted_vector
            local_weighted_vector_id = case_dict["case number"]
            local_weighted_vector_case_instructions = case_dict["Prompt Item 1"] + case_dict["Prompt Item 2"] + case_dict["Instructions (Prompt Item 3)"]
            local_weighted_vector_prompt_4 = case_dict["Prompt Item 4"]
            local_weighted_vector_prompt_3 = case_dict["Instructions (Prompt Item 3)"]

    # print(local_weighted_vector_case_instructions)
    print(f"score = {local_weighted_vector} with {local_weighted_vector_id}")
    prompt = f"[INST]\{local_weighted_vector_case_instructions}\n{ind_text}\n{local_weighted_vector_prompt_4}\n[/INST]\n\n"
    prompt_input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    words_in_prompt = len(prompt.split())
    try:
            # Here, batch size is effectively 1 since we are generating from one prompt at a time
        output = model.generate( prompt_input_ids,
                                max_new_tokens=80,  # Generate a local_cosineimum of 20 new tokens beyond the input length
                                do_sample=True,
                                temperature=0.7,
                                repetition_penalty=1.1,
                                top_p=0.7,
                                top_k=50,
                                num_return_sequences=1)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = output_text.replace(prompt, "")
    except RuntimeError as e:
        output_text = (f"Runtime error: {e}")

    # get normalized similarity score agaisnt mondo disease names
    match_pass_or_fail = string_match.count_matches(answer)
    # print(f"output: {output_text}\nanswer: {answer}\ndisease_name:{disease_name}")
    output_dict = {
        "prompt": prompt,
        "ind_text_id": ind_num,
        "ind_text": ind_text,
        "CBR+LLM Output": answer,
        "Instructions (Prompt Item 3)": local_weighted_vector_prompt_3,
        "Case ID": local_weighted_vector_id,
        "CBR_LLM Score: ": match_pass_or_fail,
    }

    print(f"output_dict: {output_dict}")
    return output_dict

if __name__ == "__main__":
    # Load the data from CSV files
    organized_ind = "/home/hs875/Llama-2/organized_indications.csv"
    cases_csv = "/home/hs875/Llama-2/cases/updated_case_base_3-31.csv"
    data = pd.read_csv(organized_ind)
    cases_data = pd.read_csv(cases_csv)

    # Pick a random row from the organized_indications.csv and extract the text excerpt
    random_row = data.sample(n=1)
    random_row_dict = random_row.to_dict(orient='records')[0]
    ind_num = random_row_dict["Number"]
    ind_text = random_row_dict["Description"]

    run_llm_with_cbr(ind_num, ind_text, cases_data)

