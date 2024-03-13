import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(59)

# Generate synthetic data for house pricing with multiple features

# Number of data points
m = 1000

# Feature 1: Size in square meters (normally distributed around 150 with a standard deviation of 30)
sizes = np.random.normal(150, 30, m)

# Feature 2: Number of bedrooms (randomly between 1 and 5)
bedrooms = np.random.randint(1, 6, m)

# Feature 3: Age of the house in years (up to 100 years)
ages = np.random.randint(0, 100, m)

# Feature 4: Number of bathrooms (randomly between 1 and 3)
bathrooms = np.random.randint(1, 4, m)

# Feature 5: Distance to city center in kilometers (normally distributed around 5 with a standard deviation of 2)
distances = np.random.normal(5, 2, m)

# Feature engineering - Interaction term: Size per bedroom
size_per_bedroom = sizes / bedrooms

# Base price for the intercept
base_price = 50000

# Assuming a simplistic linear model for the price with some noise
prices = (sizes * 3000 +
          bedrooms * 10000 +
          ages * (-200) +
          bathrooms * 15000 +
          distances * (-1000) +
          size_per_bedroom * 2000 +
          base_price +
          np.random.normal(0, 25000, m))  # Adding some noise


# Create a DataFrame
df = pd.DataFrame({
    'Size': sizes,
    'Bedrooms': bedrooms,
    'Age': ages,
    'Bathrooms': bathrooms,
    'DistanceToCityCenter': distances,
    'SizePerBedroom': size_per_bedroom,
    'Price': prices
})

# Save to CSV
csv_file_path = 'house_prices_multivariate.csv'
df.to_csv(csv_file_path, index=False)

print(f"Dataset with 1000 data points saved to {csv_file_path}")

