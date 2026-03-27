# test_analysis.py
import pandas as pd
import os

# Create a 'results' directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Generate a dummy dataset
print("Generating dummy data...")
df = pd.DataFrame({'Time': [1, 2, 3], 'Value': [10, 20, 30]})

# Save it to the results folder
output_file = 'results/dummy_output.csv'
df.to_csv(output_file, index=False)
print(f"Successfully saved test data to {output_file}")