import pandas as pd

# Load the CSV file
df = pd.read_csv('analysis_results_final.csv')

# Get all unique question types to iterate over
question_types = df['question_type'].unique()

r_squared_by_type = {}

# 1. Calculate R^2 for each question type (Stratified Analysis)
for q_type in question_types:
    # Filter the DataFrame for the current question type
    subset_df = df[df['question_type'] == q_type].copy()

    # The calculation only proceeds if there is more than 1 data point
    if len(subset_df) > 1:
        # Calculate the Pearson correlation coefficient (R)
        # using 'attention_entropy' and 'token_confidence'
        correlation = subset_df['attention_entropy'].corr(subset_df['token_confidence'])

        # Calculate R-squared (R^2)
        r_squared = correlation**2
    else:
        # Assign NaN or 0 for insufficient data points
        r_squared = float('nan')

    # Store the result
    r_squared_by_type[q_type] = r_squared

# 2. Format and Display Results
results_df = pd.DataFrame(r_squared_by_type.items(), columns=['Question Type', 'R^2'])

# Add the sample size (N) for context, as this is crucial for interpreting R^2
question_type_counts = df['question_type'].value_counts().reset_index()
question_type_counts.columns = ['Question Type', 'N']

# Merge R^2 results with counts
results_df = pd.merge(results_df, question_type_counts, on='Question Type')

# Sort by R^2 value for presentation
results_df = results_df.sort_values(by='R^2', ascending=False, na_position='last').reset_index(drop=True)

# Print the final table
print("R-squared (R^2) values by Question Type (Stratified Analysis):")
print(results_df)

# --- Optional: Calculate Overall R^2 for comparison ---
overall_correlation = df['attention_entropy'].corr(df['token_confidence'])
overall_r_squared = overall_correlation**2
print(f"\nOverall R-squared (R^2) for all data: {overall_r_squared:.4f}")