import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('final_full_settransformer_history_seed1.csv')

# Plot train_loss over epoch
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train_loss'], marker='o', linestyle='-', color='b')

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Training Loss Over Epochs')

# Add grid for better readability
plt.grid(True)

# Save the plot to a file instead of showing it
plt.savefig('training_loss_plot.png')
print("Plot saved as 'training_loss_plot.png'")