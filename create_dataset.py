import pandas as pd
from sklearn.datasets import load_digits

# Load small digits dataset (built-in)
digits = load_digits()

X = digits.data
y = digits.target

# Convert to DataFrame
df = pd.DataFrame(X)
df.insert(0, "label", y)

# Save small dataset
df.to_csv("mnist_small.csv", index=False)

print("Small dataset created: mnist_small.csv")
