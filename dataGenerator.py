import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

class DatasetProcessor:
    def __init__(self, root_dir: str, train_size: float = 0.8):
        """
        Initialize the dataset processor.

        Args:
            root_dir: Root directory containing the dataset
            train_size: Proportion of data to use for training (default: 0.8)
        """
        self.root_dir = root_dir
        self.train_size = train_size
        self.data = []

    def collect_data(self) -> None:
        """Collect data from the directory structure."""
        for path in os.listdir(self.root_dir):
            # Split path into ID and species name (format: "number. species_name")
            try:
                genus_id, species_name = path.split('. ')

                # Walk through all files in the species directory
                species_path = os.path.join(self.root_dir, path)
                for root, _, files in os.walk(species_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        self.data.append({
                            'Fname': full_path,
                            'Genera': genus_id,
                            'Species': species_name
                        })
            except ValueError:
                print(f"Skipping invalid path format: {path}")

    def create_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create training and validation datasets.

        Returns:
            Tuple containing training and validation DataFrames
        """
        # Convert collected data to DataFrame
        df = pd.DataFrame(self.data)

        # Split the data while preserving species distribution
        train_df, val_df = train_test_split(
            df,
            train_size=self.train_size,
            stratify=df['Species'],
            random_state=42
        )

        return train_df, val_df

    def process_and_save(self, output_dir: str = '.') -> None:
        """
        Process the dataset and save train/validation splits.

        Args:
            output_dir: Directory to save the output files (default: current directory)
        """
        # Collect and process data
        self.collect_data()
        train_df, val_df = self.create_datasets()

        # Save to CSV files
        train_df.to_csv(os.path.join(output_dir, 'trainData_Wingbeats.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'valiData_Wingbeats.csv'), index=False)

        # Print summary
        print(f"Dataset Summary:")
        print(f"Total samples: {len(self.data)}")
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print("\nSamples per species:")
        print(train_df['Species'].value_counts())

# Example usage
if __name__ == "__main__":
    # Initialize and run the processor
    processor = DatasetProcessor(
        root_dir=r"D:\ICS Project\Project\WINGBEATS",
        train_size=0.8
    )
    processor.process_and_save(output_dir=r"D:\ICS Project\Project\Output")
