import pandas as pd
import os

# Define the set of experiment, model set, and time
set_of_exp = "nj_3000"
model_set = "mistral-nemo"
time = "closest"

# Define the range of parts to process
parts_range = range(1, 9)  # From part 1 to part 6

# Initialize an empty DataFrame to store all output data
all_output_data = []

for part in parts_range:
    # Load additional data for the current part
    additional_data_path = rf"E:\{model_set}\{time}\{set_of_exp}\{set_of_exp}_{time}_{model_set}_part_{part}.csv"
    additional_data = pd.read_csv(additional_data_path)

    # Define the user files path
    user_files_path = rf"E:\{set_of_exp}_{time}_original"

    # Initialize a list to store output data for the current part
    output_data = []

    # Iterate over each user file
    for filename in os.listdir(user_files_path):
        if filename.endswith(".csv"):
            # Extract user ID from the filename
            user_id = filename.split('_')[1].split('.')[0]

            # Load user data
            user_data = pd.read_csv(os.path.join(user_files_path, filename))

            # Extract text data
            text_data = ' '.join(user_data['text'].tolist()).replace('\n', ' ').replace('\r', ' ')

            # Check if matching user_id is found
            user_additional_data = additional_data[additional_data['user_id'] == int(user_id)]
            if user_additional_data.empty:
                print(f"No additional data found for user_id: {user_id}")
                continue

            # Get the matching row
            user_additional_data = user_additional_data.iloc[0]
            tone_of_voice = user_additional_data['tone_of_voice']
            attitude = user_additional_data['attitude']
            predicted_content = user_additional_data['predicted_content']
            address = user_additional_data['address']

            # Format the output text
            #NJ
            output_text = (
                f"Hurricane Oscar just made landfall near Atlantic City. A resident in New Jersey who is currently at {address} with a tone of voice of {tone_of_voice} on social media is {attitude} about this hurricane. "
                f"This user composed a new comment: {predicted_content}. "
                f"With your professional knowledge in psychology, social science, and behavioral science, please analyze if the language use and the enunciated attitude of this new comment conforms with that of the previous ones. "
                f"The previous social media comments are provided as follows: {text_data}."
            )
            #NY
            # output_text = (
            #     f"Hurricane Oscar just made landfall 160km south of Long Island as a Category 1 hurricane. A resident in Long Island who is currently at {address} with a tone of voice of {tone_of_voice} on social media is {attitude} about this hurricane."
            #     f"This user composed a new comment: {predicted_content}, "
            #     f"With your professional knowledge in psychology, social science, and behavorial science, please analyze if the language use and the enunciated attitude of this new comment conforms with that of the previous ones."
            #     f"The previous social media comments are provided as follows: {text_data}."
            # )

            # Clean output_text
            output_text = output_text.replace('\n', ' ').replace('\r', ' ')

            # Add to output data
            output_data.append([user_id, output_text])

    # Convert output data to DataFrame
    output_df = pd.DataFrame(output_data, columns=['user_id', 'output_text'])

    # Save the output data for the current part
    output_df.to_csv(rf"E:\{model_set}\{time}\{set_of_exp}_{time}_{model_set}_part_{part}.csv", index=False)
    all_output_data.append(output_df)

print("Processed files saved.")
