import pandas as pd
from typing import Dict, List, Optional, Union

def validate_and_rename_columns(
    df: pd.DataFrame,
    case_sensitive: bool = False
) -> pd.DataFrame:
    """
    Validates and renames columns in a DataFrame to match required column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame with columns to validate and potentially rename
        required_columns (Dict[str, str]): Dictionary mapping current column names to required names
            e.g., {'Text': 'Resume'} will rename 'Text' column to 'Resume'
        case_sensitive (bool): Whether to perform case-sensitive column matching
            
    Returns:
        pd.DataFrame: DataFrame with validated and renamed columns
        
    Raises:
        ValueError: If required columns are missing and no matching columns are found
    """
    required_columns = {
        'Text': 'Resume',  # Rename 'Text' to 'Resume'
        'Category': 'Category'  # Keep 'Category' as is
        }
    
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Get current column names
    current_columns = df_copy.columns.tolist()
    
    # Convert column names to lowercase if case-insensitive matching
    if not case_sensitive:
        column_map = {col.lower(): col for col in current_columns}
        required_map = {old.lower(): new for old, new in required_columns.items()}
    else:
        column_map = {col: col for col in current_columns}
        required_map = required_columns
    
    # Track missing columns
    missing_columns = []
    
    # Perform column validation and renaming
    for old_name, new_name in required_columns.items():
        old_name_check = old_name if case_sensitive else old_name.lower()
        
        if old_name_check in column_map:
            # Rename column if it exists
            if old_name != new_name:
                df_copy = df_copy.rename(columns={column_map[old_name_check]: new_name})
        else:
            missing_columns.append(old_name)
    
    if missing_columns:
        raise ValueError(
            f"Required columns {missing_columns} not found in DataFrame. "
            f"Available columns: {current_columns}"
        )
    
    return df_copy


import matplotlib.pyplot as plt


class PlotMetrics:
    """Class for generating plots of various model evaluation metrics"""
    
    def __init__(self, save_dir=None):
        """
        Initialize PlotMetrics class
        
        Args:
            save_dir (str, optional): Directory to save plots. Defaults to None.
        """
        self.save_dir = save_dir
        
    def plot_accuracy(self, history):
        """Plot training and validation accuracy"""
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        if self.save_dir:
            plt.savefig(f'{self.save_dir}/accuracy.png')
        plt.show()
        
    def plot_loss(self, history):
        """Plot training and validation loss"""
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        if self.save_dir:
            plt.savefig(f'{self.save_dir}/loss.png')
        plt.show()
        
    def plot_precision(self, history):
        """Plot training and validation precision"""
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['precision'], label='Training Precision')
        plt.plot(history.history['val_precision'], label='Validation Precision')
        plt.title('Model Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        if self.save_dir:
            plt.savefig(f'{self.save_dir}/precision.png')
        plt.show()
        
    def plot_recall(self, history):
        """Plot training and validation recall"""
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['recall'], label='Training Recall')
        plt.plot(history.history['val_recall'], label='Validation Recall')
        plt.title('Model Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        if self.save_dir:
            plt.savefig(f'{self.save_dir}/recall.png')
        plt.show()
        
    def plot_auc(self, history):
        """Plot training and validation AUC"""
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['AUC'], label='Training AUC')
        plt.plot(history.history['val_AUC'], label='Validation AUC')
        plt.title('Model AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        if self.save_dir:
            plt.savefig(f'{self.save_dir}/auc.png')
        plt.show()

    def plot_combined_metrics(self, history):
        """Plot all metrics (accuracy, loss, precision, recall, AUC) in a single figure"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Model Metrics Overview', fontsize=16)
        
        # Plot accuracy
        axes[0,0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0,0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0,0].set_title('Model Accuracy')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].legend()

        # Plot loss
        axes[0,1].plot(history.history['loss'], label='Training Loss')
        axes[0,1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0,1].set_title('Model Loss')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()

        # Plot precision
        axes[1,0].plot(history.history['precision'], label='Training Precision')
        axes[1,0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1,0].set_title('Model Precision')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].legend()

        # Plot recall
        axes[1,1].plot(history.history['recall'], label='Training Recall')
        axes[1,1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1,1].set_title('Model Recall')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Recall')
        axes[1,1].legend()

        # Plot AUC
        axes[2,0].plot(history.history['AUC'], label='Training AUC')
        axes[2,0].plot(history.history['val_AUC'], label='Validation AUC')
        axes[2,0].set_title('Model AUC')
        axes[2,0].set_xlabel('Epoch')
        axes[2,0].set_ylabel('AUC')
        axes[2,0].legend()

        # Adjust layout
        plt.tight_layout()
        
        if self.save_dir:
            plt.savefig(f'{self.save_dir}/combined_metrics.png')
        plt.show()