import pandas as pd
import requests
import numpy as np
import os

set_of_exp = "nj_3000"
model_set = "gemma2"
time = "closest"
part = "6" #distributed training
# Local file path
input_csv_path = rf"Your_input_file_path"
output_directory =rf"The_output_file_path"
output_csv_path = os.path.join(output_directory, f'{set_of_exp}_{time}_{model_set}_part_{part}.csv')

# Remote API address
remote_api_url = "http://xx.xxx.xx.xx:5006/api/chat"

# Read the input CSV file
try:
    input_df = pd.read_csv(input_csv_path, encoding='latin1')
except UnicodeDecodeError:
    print("Failed to read the CSV file with 'latin1' encoding.")
    exit(1)

# Initialize the output list
responses = []

# Walk through each line, extract the output text and call the remote model
for index, row in input_df.iterrows():
    output_text = row['output_text']

    # Clean up the data to make sure there are no floating point values out of range
    if isinstance(output_text, float) and (np.isnan(output_text) or np.isinf(output_text)):
        output_text = ""

    # Construct request data
    request_data = {
        "text": output_text
    }

    # Send the request to the remote server and get the response
    response = requests.post(remote_api_url, json=request_data)
    if response.status_code == 200:
        response_text = response.json().get("response", "")
        print(response_text)
    else:
        response_text = "Error: Unable to get response from server."

    response_text = response_text.replace('\n', ' ').replace('\r', ' ')
    # Add the response to the list
    responses.append(response_text)

# Adds the response to a new column of the DataFrame
input_df['response'] = responses

os.makedirs(output_directory, exist_ok=True)
# Save the DataFrame to a CSV file
input_df.to_csv(output_csv_path, index=False)

print(f"Responses saved to {output_csv_path}")
