import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df = pd.read_csv('analysis_results_final.csv')

print("="*80)
print("DIAGNOSTIC ANALYSIS OF ATTENTION RESULTS")
print("="*80)

# 1. Entropy Analysis
print("\n1. ENTROPY DISTRIBUTION")
print("-"*40)
print(f"Attention Entropy - Mean: {df['attention_entropy'].mean():.4f}")
print(f"Attention Entropy - Std:  {df['attention_entropy'].std():.4f}")
print(f"Attention Entropy - Min:  {df['attention_entropy'].min():.4f}")
print(f"Attention Entropy - Max:  {df['attention_entropy'].max():.4f}")
print(f"\nToken Confidence - Mean: {df['token_confidence'].mean():.4f}")
print(f"Token Confidence - Std:  {df['token_confidence'].std():.4f}")

# Check if entropy is essentially constant
if df['attention_entropy'].std() < 0.01:
    print("\n⚠️  WARNING: Attention entropy shows almost no variation!")
    print("   This suggests attention is uniformly dispersed across all images.")
    
if df['token_confidence'].std() < 0.01:
    print("\n⚠️  WARNING: Token entropy is constant!")
    print("   This suggests the model has maximum uncertainty about predictions.")

# 2. Cluster Analysis
print("\n2. CLUSTER STATISTICS")
print("-"*40)
print(f"Average clusters per image: {df['n_clusters'].mean():.2f}")
print(f"Average noise points: {df['n_noise'].mean():.2f}")
print(f"Images with 0 clusters: {(df['n_clusters'] == 0).sum()}")

# Calculate noise ratio
total_points = 576  # 24x24 grid after processing
df['noise_ratio'] = df['n_noise'] / total_points
print(f"\nAverage noise ratio: {df['noise_ratio'].mean():.2%}")
print(f"Median noise ratio: {df['noise_ratio'].median():.2%}")

if df['noise_ratio'].mean() > 0.5:
    print("\n⚠️  WARNING: Over 50% of attention points are classified as noise!")
    print("   Consider:")
    print("   - Reducing DBSCAN eps parameter (currently 1.3)")
    print("   - Reducing min_samples (currently 15)")
    print("   - The attention signal may be too weak/dispersed")

# 3. Question Type Analysis
print("\n3. ANALYSIS BY QUESTION TYPE")
print("-"*40)
question_type_stats = df.groupby('question_type').agg({
    'attention_entropy': 'mean',
    'n_clusters': 'mean',
    'noise_ratio': 'mean'
}).round(4)
print(question_type_stats)

# 4. Look for patterns
print("\n4. CORRELATION ANALYSIS")
print("-"*40)
correlations = df[['n_clusters', 'n_noise', 'attention_entropy', 'noise_ratio']].corr()
print(correlations)

# 5. Check if results match paper expectations
print("\n5. COMPARISON TO PAPER EXPECTATIONS")
print("-"*40)
print("According to the paper:")
print("- R² between attention and confidence should be < 0.15")
print("- Different question types should show different patterns")
print("- Counting questions should have higher R² (up to 0.28)")
print("\nYour results:")
if df['attention_entropy'].std() < 0.01:
    print("❌ All entropy values are essentially identical")
    print("   This prevents meaningful correlation analysis")
else:
    print("✓ Entropy shows variation across images")

# 6. Identify problematic images
print("\n6. PROBLEMATIC IMAGES")
print("-"*40)
no_clusters = df[df['n_clusters'] == 0]
if len(no_clusters) > 0:
    print(f"\n{len(no_clusters)} images with NO clusters detected:")
    for idx, row in no_clusters.head(10).iterrows():
        print(f"  - Image {row['image_id']}: {row['question_type']} - {row['question'][:50]}...")

high_noise = df[df['noise_ratio'] > 0.8]
if len(high_noise) > 0:
    print(f"\n{len(high_noise)} images with >80% noise:")
    for idx, row in high_noise.head(5).iterrows():
        print(f"  - Image {row['image_id']}: {row['question_type']} - noise ratio: {row['noise_ratio']:.2%}")

# 7. Recommendations
print("\n7. RECOMMENDATIONS")
print("-"*40)
print("\n🔧 IMMEDIATE FIXES:")
print("1. Simplify prompts - remove redundant instructions")
print("   Current:  'Count exactly how many cats. How many cats? Count:'")
print("   Better:   'How many cats are in the image?'")
print("   Prefix:   'There are'")
print("\n2. Check attention extraction:")
print("   - Verify the correct head/layer is being selected")
print("   - Print out raw attention values before normalization")
print("   - Ensure attention isn't being flattened/averaged incorrectly")
print("\n3. Adjust DBSCAN parameters:")
print("   - Try eps=1.0 (currently 1.3)")
print("   - Try min_samples=10 (currently 15)")
print("\n4. Verify model predictions:")
print("   - Check if the model is actually generating correct answers")
print("   - High token entropy suggests model uncertainty")

# Generate visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Entropy distribution
axes[0, 0].hist(df['attention_entropy'], bins=30, edgecolor='black')
axes[0, 0].set_title('Attention Entropy Distribution')
axes[0, 0].set_xlabel('Entropy')
axes[0, 0].set_ylabel('Frequency')

# Plot 2: Clusters vs Noise
axes[0, 1].scatter(df['n_clusters'], df['n_noise'], alpha=0.5)
axes[0, 1].set_title('Clusters vs Noise Points')
axes[0, 1].set_xlabel('Number of Clusters')
axes[0, 1].set_ylabel('Number of Noise Points')

# Plot 3: Clusters by question type
question_clusters = df.groupby('question_type')['n_clusters'].mean().sort_values()
axes[1, 0].barh(range(len(question_clusters)), question_clusters.values)
axes[1, 0].set_yticks(range(len(question_clusters)))
axes[1, 0].set_yticklabels(question_clusters.index, fontsize=8)
axes[1, 0].set_title('Average Clusters by Question Type')
axes[1, 0].set_xlabel('Average Clusters')

# Plot 4: Noise ratio by question type
question_noise = df.groupby('question_type')['noise_ratio'].mean().sort_values()
axes[1, 1].barh(range(len(question_noise)), question_noise.values)
axes[1, 1].set_yticks(range(len(question_noise)))
axes[1, 1].set_yticklabels(question_noise.index, fontsize=8)
axes[1, 1].set_title('Average Noise Ratio by Question Type')
axes[1, 1].set_xlabel('Noise Ratio')

plt.tight_layout()
plt.savefig('diagnostic_analysis_topk.png', dpi=150, bbox_inches='tight')
print("\n📊 Visualization saved as 'diagnostic_analysis.png'")