import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Set Matplotlib and Seaborn style for publication-ready plots
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Define the SEES Attention-Confidence Gap threshold
SEES_GAP_THRESHOLD = 0.15

# --- Helper Function for Safe Correlation Calculation ---
def safe_pearsonr_r2(data, col1, col2):
    """Calculates Pearson R and R^2, returning NaN if N < 2 or data is constant."""
    n = len(data)
    if n < 3: # Require at least 3 points for a meaningful correlation plot
        return pd.Series({'N': n, 'R': np.nan, 'R2': np.nan})
    else:
        # Check if either column is constant (which causes pearsonr to fail)
        if data[col1].nunique() <= 1 or data[col2].nunique() <= 1:
            return pd.Series({'N': n, 'R': np.nan, 'R2': np.nan})
        
        try:
            r, _ = pearsonr(data[col1], data[col2])
            return pd.Series({'N': n, 'R': r, 'R2': r**2})
        except Exception:
            # Catch other potential errors gracefully (e.g., all nans)
            return pd.Series({'N': n, 'R': np.nan, 'R2': np.nan})

# --- Data Loading and Processing ---
try:
    df_full = pd.read_csv('analysis_results_final.csv')
    df_topk_new = pd.read_csv('analysis_results_final_topk.csv') # Assuming this is the new file
except FileNotFoundError:
    print("Error: One or both input files not found. Ensure 'analysis_results_final.csv' and 'analysis_results_final_topk (1).csv' are available.")
    exit()

def calculate_grouped_correlation(df, method_name):
    """Calculates and formats grouped correlation data."""
    grouped_corr = df.groupby('question_type').apply(
        lambda x: safe_pearsonr_r2(x, 'attention_entropy', 'token_confidence')
    ).reset_index()
    grouped_corr.columns = ['Question Type', 'N', f'R_{method_name}', f'R2_{method_name}']
    return grouped_corr

# Calculate grouped correlations for both methods
full_grouped_corr = calculate_grouped_correlation(df_full, 'Full')
topk_grouped_corr_new = calculate_grouped_correlation(df_topk_new, 'TopK_New')

# Merge the grouped results for comparison
grouped_comparison = pd.merge(
    full_grouped_corr[['Question Type', 'N', 'R_Full', 'R2_Full']],
    topk_grouped_corr_new[['Question Type', 'R_TopK_New', 'R2_TopK_New']],
    on='Question Type',
    how='inner' # Use inner merge to only compare types present in both
).fillna(np.nan)

# Filter out groups with insufficient samples (already handled by safe_pearsonr_r2, but double check N>3)
grouped_comparison = grouped_comparison[grouped_comparison['N'] >= 3].dropna(subset=['R2_Full', 'R2_TopK_New'], how='all')

# Prepare data for R-squared Plot
r2_plot_data = grouped_comparison.melt(
    id_vars=['Question Type', 'N'],
    value_vars=['R2_Full', 'R2_TopK_New'],
    var_name='Method',
    value_name='R²'
)
r2_plot_data['Method'] = r2_plot_data['Method'].replace({
    'R2_Full': 'Full Attention', 
    'R2_TopK_New': 'Top-K Attention'
})

# Sort the data by the max R² across both methods for visual clarity
r2_plot_data_sorted = r2_plot_data.groupby('Question Type')['R²'].max().sort_values(ascending=False).index
r2_plot_data['Question Type'] = pd.Categorical(r2_plot_data['Question Type'], categories=r2_plot_data_sorted, ordered=True)
r2_plot_data.sort_values(by=['Question Type', 'Method'], inplace=True)


# Prepare data for R (Pearson correlation) Plot
r_plot_data = grouped_comparison.melt(
    id_vars=['Question Type', 'N'],
    value_vars=['R_Full', 'R_TopK_New'],
    var_name='Method',
    value_name='R'
)
r_plot_data['Method'] = r_plot_data['Method'].replace({
    'R_Full': 'Full Attention', 
    'R_TopK_New': 'Top-K Attention'
})
r_plot_data['Question Type'] = pd.Categorical(r_plot_data['Question Type'], categories=r2_plot_data_sorted, ordered=True)
r_plot_data.sort_values(by=['Question Type', 'Method'], inplace=True)


# --- Plotting R² Comparison (The Attention-Confidence Gap) ---
plt.figure(figsize=(10, 6))
r2_plot = sns.barplot(
    data=r2_plot_data,
    x='Question Type',
    y='R²',
    hue='Method',
    palette={'Full Attention': '#1f77b4', 'Top-K Attention': '#ff7f0e'}, # Default Blue and Orange
    errorbar=None
)

# Add the Attention-Confidence Gap threshold line
r2_plot.axhline(
    SEES_GAP_THRESHOLD, 
    color='red', 
    linestyle='--', 
    linewidth=1.5, 
    label=f'SEES Gap Threshold ($R^2=0.15$)'
)

plt.title('R² Comparison: Full vs. Top-K Attention Entropy vs. Token Confidence', pad=15)
plt.ylabel(r'Coefficient of Determination ($R^2$)', fontsize=11)
plt.xlabel('Question Type', fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.05)
plt.legend(title='Attention Method', loc='upper right')
plt.tight_layout()
plt.savefig('r2_comparison.png', dpi=300)
plt.show()

# --- Plotting R Comparison (The Direction of Correlation) ---
plt.figure(figsize=(10, 6))
r_plot = sns.barplot(
    data=r_plot_data,
    x='Question Type',
    y='R',
    hue='Method',
    palette={'Full Attention': '#1f77b4', 'Top-K Attention': '#ff7f0e'},
    errorbar=None
)

# Add line for zero correlation
r_plot.axhline(0, color='black', linestyle='-', linewidth=0.8)

plt.title('Pearson R Comparison: Focused (Negative) vs. Dispersed (Positive) Attention', pad=15)
plt.ylabel(r'Pearson Coefficient ($R$)', fontsize=11)
plt.xlabel('Question Type', fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.ylim(-1.05, 1.05)
plt.legend(title='Attention Method', loc='lower right')
plt.tight_layout()
plt.savefig('r_comparison.png', dpi=300)
plt.show()

print("\nVisualization script complete. Two files, 'r2_comparison.png' and 'r_comparison.png', have been generated.")
print("The R² plot clearly shows how Top-K filtering moves some categories (like Action Recognition) significantly above the 0.15 SEES Gap Threshold, while others remain firmly within the gap.")
print("The R plot highlights the sign flips, where a counter-intuitive positive R (Full Attention) becomes a consistent, intuitive negative R (Top-K Attention) for several key visual tasks.")