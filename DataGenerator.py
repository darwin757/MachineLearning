import pandas as pd
import numpy as np

# Generate random house sizes and prices
np.random.seed(59) # for reproducible results
sizes = np.random.randint(50, 500, 1000) # sizes between 50 and 250 square meters
prices = sizes * 1000 + np.random.randint(-50000, 500000, 1000) # base price plus some noise

# Create a DataFrame and save to CSV
df = pd.DataFrame({'Size': sizes, 'Price': prices})
df.to_csv('house_prices.csv', index=False)
