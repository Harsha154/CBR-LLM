import torch
import pandas as pd
import re
import mondo_similarity
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize the device and model
torch.cuda.set_device(1)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/Llama-2-7B-32K-Instruct")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/Llama-2-7B-32K-Instruct",
                                             trust_remote_code=True, torch_dtype=torch.float16).to(device)

def run_llm_with_no_cbr(ind_num, ind_text, prompt_id):
    # Tokenize and encode the text excerpt to get its embedding
    input_ids = tokenizer.encode(ind_text, return_tensors="pt").to(device)
    with torch.no_grad():
        model_output = model(input_ids, output_hidden_states=True)
        input_embedding = model_output.hidden_states[-1].mean(dim=1)  # Average pooling over the sequence
    
    input_text_length = len(ind_text)

    Prompt1 = "what is the disease described in this text? Limit answer to the disease name."
    Prompt4 = "Find the disease and format your answer like this: [disease]. If there is more than one disease, then format your answer like this: [disease 1, disease 2, etc.]. Do not include any other information."
            
    prompt = f"[INST]\{Prompt1}\n{ind_text}\n{Prompt4}\n[/INST]\n\n"
    prompt_input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    # words_in_prompt = len(prompt.split())
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
        answer = (f"Runtime error: {e}")
    
    # get normalized similarity score agaisnt mondo disease names
    norm_similarity_score_dict = mondo_similarity.get_similary_score_with_mondo(answer, prompt_id)
    # print(f"output: {output_text}\nanswer: {answer}\ndisease_name:{disease_name}")
    output_dict = {
        "ind_text": ind_num,
        "CBR+LLM Output": answer,
        "LLM_NO_CBR Score: ": norm_similarity_score_dict,
    }
    print(f"output_dict: {output_dict}")
    return output_dict

if __name__ == "__main__":
    # Load the data from CSV files
    organized_ind = "organized_indications.csv"
    cases_csv = "updated_case_base_3-31.csv"
    data = pd.read_csv(organized_ind)
    cases_data = pd.read_csv(cases_csv) 
    prompt_id = 00000

    # Pick a random row from the organized_indications.csv and extract the text excerpt
    random_row = data.sample(n=1)
    random_row_dict = random_row.to_dict(orient='records')[0]
    ind_num = random_row_dict["Number"]
    ind_text = random_row_dict["Description"]

    run_llm_with_no_cbr(ind_num, ind_text, prompt_id)
