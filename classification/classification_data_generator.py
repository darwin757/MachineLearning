import pandas as pd
from sklearn.datasets import make_classification

# Generate a dataset for tumor classification
def generate_tumor_dataset(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, n_classes=2, random_state=42):
    """
    Generate a tumor classification dataset and save it to a CSV file.

    Parameters:
    - n_samples: The number of samples.
    - n_features: The total number of features.
    - n_informative: The number of informative features.
    - n_redundant: The number of redundant features.
    - n_classes: The number of classes (or labels) in the dataset.
    - random_state: The seed used by the random number generator.
    """
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                               n_redundant=n_redundant, n_classes=n_classes, random_state=random_state)
    
    # Convert the dataset to a DataFrame
    df_features = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(n_features)])
    df_target = pd.DataFrame(y, columns=["Target"])
    df = pd.concat([df_features, df_target], axis=1)
    
    # Save the dataset to a CSV file
    df.to_csv("tumor_classification_dataset.csv", index=False)
    print("Dataset generated and saved to tumor_classification_dataset.csv")

# Generate and save the dataset
generate_tumor_dataset()
