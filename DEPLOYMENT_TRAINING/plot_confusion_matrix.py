import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the confusion matrix CSV
cm_df = pd.read_csv('final_full_settransformer_cm_train_seed1.csv')

# The CSV has columns 0,1,2,3,4,5 as headers, and rows are the matrix
# Convert to numpy array
cm = cm_df.values

# Number of classes
num_classes = cm.shape[0]

# Calculate precision, recall, f1 for each class
precision = np.zeros(num_classes)
recall = np.zeros(num_classes)
f1 = np.zeros(num_classes)

for i in range(num_classes):
    tp = cm[i, i]
    fp = np.sum(cm[:, i]) - tp
    fn = np.sum(cm[i, :]) - tp
    tn = np.sum(cm) - tp - fp - fn

    precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

# Overall macro averages
overall_precision = np.mean(precision)
overall_recall = np.mean(recall)
overall_f1 = np.mean(f1)

# Plot confusion matrix
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Pred {i}' for i in range(num_classes)],
            yticklabels=[f'True {i}' for i in range(num_classes)])

plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add metrics as text on the plot - positioned in the top-right corner
metrics_text = f'Overall F1 (Macro): {overall_f1:.3f}\nOverall Precision (Macro): {overall_precision:.3f}\nOverall Recall (Macro): {overall_recall:.3f}'
plt.figtext(0.85, 0.15, metrics_text, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Save the plot
plt.tight_layout()
plt.savefig('confusion_matrix_plot.png', dpi=300, bbox_inches='tight')
print("Confusion matrix plot saved as 'confusion_matrix_plot.png'")