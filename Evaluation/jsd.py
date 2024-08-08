import pandas as pd
import numpy as np

def sentiment_mapping(df):
    # Mapping for polarity
    df['polarity'] = df['sentiment'].map({0: -1, 1: -1, 2: 0, 3: 1, 4: 1})
    df['gt_polarity'] = df['gt'].map({0: -1, 1: -1, 2: 0, 3: 1, 4: 1})
    return df

def calculate_pdf_t(df, column):
    total_count = df[column].count()
    value_counts = df[column].value_counts(normalize=True)
    pdf = value_counts.reindex([-1, 0]).fillna(0) / total_count
    return pdf

def calculate_pdf_s(df, column):
    total_count = df[column].count()
    value_counts = df[column].value_counts(normalize=True)
    pdf = value_counts.reindex([0, 1, 2, 3, 4]).fillna(0) / total_count
    return pdf

def calculate_pdf_p(df, column):
    total_count = df[column].count()
    value_counts = df[column].value_counts(normalize=True)
    pdf = value_counts.reindex([-1, 0, 1]).fillna(0) / total_count
    return pdf

def calculate_pdf_m(df, column):
    total_count = df[column].count()
    value_counts = df[column].value_counts(normalize=True)
    pdf = value_counts.reindex([0, 1, 2]).fillna(0) / total_count
    return pdf

def shannon_entropy(pdf):
    return -np.sum(pdf * np.log2(pdf.replace(0, np.nan)))


def js_divergence(pdf1, pdf2):
    m = 0.5 * (pdf1 + pdf2)
    print(m)
    return shannon_entropy(m) - 0.5 * (shannon_entropy(pdf1) + shannon_entropy(pdf2))


def analyze_sentiments(df, gt_col, pred_col):
    df['sentiment'] = df[pred_col]
    df = sentiment_mapping(df)

    pdf_sentiment_gt = calculate_pdf_s(df, gt_col)
    pdf_sentiment_pred = calculate_pdf_s(df, 'sentiment')
    jsd_sentiment = js_divergence(pdf_sentiment_gt, pdf_sentiment_pred)

    pdf_polarity_gt = calculate_pdf_p(df, 'gt_polarity')
    pdf_polarity_pred = calculate_pdf_p(df, 'polarity')
    jsd_polarity = js_divergence(pdf_polarity_gt, pdf_polarity_pred)

    return jsd_sentiment, jsd_polarity

def calculate_jsd_for_columns(df, column_pairs):
    results = {}
    for gt_col, pred_col in column_pairs:
        jsd_sentiment, jsd_polarity, jsd_twofold = analyze_sentiments(df, gt_col, pred_col)
        results[(gt_col, pred_col, 'sentiment')] = jsd_sentiment
        results[(gt_col, pred_col, 'polarity')] = jsd_polarity
        results[(gt_col, pred_col, 'towfold')] = jsd_twofold
    return results

# Define the column pairs to be evaluated
column_pairs = [
    ('gt', 'XTONE'),('gt', 'SPR'),('gt', 'FTP'),('gt','PREDSEN')
]

set_of_exp = "nj_3000"
time = 'after'
model_set = "gemma2"


data_path = rf'E:\{set_of_exp}_{time}_{model_set}.csv'
data = pd.read_csv(data_path)

# Calculate JSD
jsd_results = calculate_jsd_for_columns(data, column_pairs)

# Print results
for columns, jsd_value in jsd_results.items():
    print(f'JSD {set_of_exp}_{time}_{model_set} between {columns[0]} and {columns[1]}: {jsd_value}')