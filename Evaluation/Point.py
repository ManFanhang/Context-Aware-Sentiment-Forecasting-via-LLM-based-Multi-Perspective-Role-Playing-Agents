import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

set_of_exp = "ny_3000"
time = 'closest'
model_set = "gemma2"

# Input and output file paths
input_file_path = rf'E:\{set_of_exp}_{time}_{model_set}.csv'
output_file_path = rf'E:\{set_of_exp}_{time}_{model_set}.csv'
# Read the CSV file
df = pd.read_csv(input_file_path)
# Define the column pairs to be evaluated
column_pairs = [
    ('gt', 'XTONE'),('gt', 'SPR'),('gt', 'FTP'),('gt','PREDSEN')
]


results = []

for col1, col2 in column_pairs:
    f1 = f1_score(df[col1], df[col2],  average='macro')
    acc = accuracy_score(df[col1], df[col2])
    results.append({
        'Column Pair': f'{col1} vs {col2}',
        'Label': 'Overall',
        'Precision': 'N/A',
        'Recall': 'N/A',
        'F1 Score': f1,
        'Accuracy': acc
    })
    for label in [0, 1,2,3,4]:
        pre = precision_score(df[col1], df[col2], labels=[label], average='macro')
        rec = recall_score(df[col1], df[col2], labels=[label], average='macro')
        f1 = f1_score(df[col1], df[col2], labels=[label], average='macro')
        results.append({
            'Column Pair': f'{col1} vs {col2}(5)',
            'Label': label,
            'Precision': pre,
            'Recall': rec,
            'F1 Score': f1,
            'Accuracy': 'N/A'
        })

input_file_path = rf'E:\{set_of_exp}_{time}_{model_set}.csv'

df = pd.read_csv(input_file_path)

# Convert to three categories and recalculate
df_replaced = df.replace({0: 1, 4: 3})

for col1, col2 in column_pairs:
    f1 = f1_score(df_replaced[col1], df_replaced[col2],  average='macro')
    acc = accuracy_score(df_replaced[col1], df_replaced[col2])
    results.append({
        'Column Pair': f'{col1} vs {col2} (3 classes)',
        'Label': 'Overall',
        'Precision': 'N/A',
        'Recall': 'N/A',
        'F1 Score': f1,
        'Accuracy': acc
    })
    for label in [1, 2, 3]:
        pre = precision_score(df_replaced[col1], df_replaced[col2], labels=[label], average='macro')
        rec = recall_score(df_replaced[col1], df_replaced[col2], labels=[label], average='macro')
        f1 = f1_score(df_replaced[col1], df_replaced[col2], labels=[label], average='macro')
        results.append({
            'Column Pair': f'{col1} vs {col2} (3 classes)',
            'Label': label,
            'Precision': pre,
            'Recall': rec,
            'F1 Score': f1,
            'Accuracy': 'N/A'
        })

# Convert the result to DataFrame
results_df = pd.DataFrame(results)

# Save the results to a new CSV file
results_df.to_csv(output_file_path, index=False)

print(f"The precision, recall, and accuracy results have been saved to {output_file_path}")