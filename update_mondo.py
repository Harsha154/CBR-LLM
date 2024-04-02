def extract_names(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split('\t')
            if len(parts) > 3 and parts[2] == 'name':
                outfile.write(parts[3].strip('\"') + '\n')  # Strip quotes and write the name

# Set the file names
input_file = 'mondo-diseases.txt'
output_file = 'names_only.txt'

# Call the function to process the file
extract_names(input_file, output_file)

# Displaying the content of the output file to confirm
with open(output_file, 'r') as file:
    extracted_names = file.read()

extracted_names

