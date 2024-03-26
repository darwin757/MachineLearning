import pandas as pd
from sklearn.datasets import make_classification

# Generate a dataset for tumor classification
def generate_tumor_dataset(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, n_classes=2, random_state=42):
    """
    Generate a tumor classification dataset with realistic feature names for the features and 
    keep the last column as the target for classification purposes. The dataset is then saved to a CSV file.

    Parameters:
    - n_samples: The number of samples.
    - n_features: The total number of features.
    - n_informative: The number of informative features.
    - n_redundant: The number of redundant features.
    - n_classes: The number of classes (or labels) in the dataset.
    - random_state: The seed used by the random number generator.
    """
    # Ensure the number of features does not exceed the length of the feature_names list minus one for the target column
    assert n_features <= 20, "n_features cannot exceed 20 for this example."
    
    # Realistic feature names (for example purposes, actual names should be determined by domain experts)
    feature_names = [
        "Cell_Size", "Cell_Shape", "Marginal_Adhesion", "Single_Epithelial_Size", 
        "Bare_Nuclei", "Bland_Chromatin", "Normal_Nucleoli", "Mitoses", 
        "Clump_Thickness", "Uniformity_of_Cell_Size", "Uniformity_of_Cell_Shape", 
        "Mitotic_Count", "Texture", "Area", "Smoothness", "Compactness", 
        "Concavity", "Concave_Points", "Symmetry", "Fractal_Dimension"
    ]

    # Generate the dataset
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                               n_redundant=n_redundant, n_classes=n_classes, random_state=random_state)
    
    # Convert the features to a DataFrame with realistic feature names
    df_features = pd.DataFrame(X, columns=feature_names[:n_features])
    
    # The target column is handled separately
    df_target = pd.DataFrame(y, columns=["Target"])
    
    # Concatenate the features and target into one DataFrame
    df = pd.concat([df_features, df_target], axis=1)
    
    # Save the dataset to a CSV file
    df.to_csv("tumor_classification_dataset.csv", index=False)
    print("Dataset generated and saved to tumor_classification_dataset.csv with realistic feature names")

# Generate and save the dataset
generate_tumor_dataset()
