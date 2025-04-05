CONFIG = {
    ### Data paths
    #
    "data_path": "data/processed/",  # Path to your preprocessed dataset
    "raw_data_path": "data/raw/",  # Optional: path for raw data files
    #
    #
    ### Model hyperparameters for N-gram model
    #
    "ngram_n": 3,  # n in n-gram (e.g., 3 for trigram)
    #
    #
    ### Training parameters common to neural models
    #
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 10,  # Number of training epochs
    "warmup_steps": 100,  # Used for transformer scheduler
    # Sequence processing
    "max_seq_length": 50,  # Maximum sequence length for input/output
    # Additional configurations for model-specific parameters
    "lstm_hidden_size": 256,  # Hidden state size for LSTM model
    "lstm_num_layers": 2,  # Number of layers for LSTM model
    "dropout": 0.5,  # Dropout rate for regularization
    #
    #
    ### Transformer specific parameters (if fine-tuning a pre-trained model)
    #
    "pretrained_model_name": "t5-small",  # Name of the pre-trained model from Hugging Face
    "transformer_max_length": 50,  # Max token length for transformer inputs/outputs
    # Logging and checkpointing
    "log_interval": 100,  # How often to log training progress (in batches)
    "checkpoint_dir": "checkpoints/",  # Directory to save model checkpoints
}

if __name__ == "__main__":
    # Quick test to print configuration values
    for key, value in CONFIG.items():
        print(f"{key}: {value}")
