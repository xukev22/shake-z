CONFIG = {
    ### Data paths
    "data_path": "data/processed/",  # Path to your preprocessed dataset
    "raw_data_path": "data/raw/",  # Path to raw data
    "dataset": "shakez",  # "shakez", "sonnetz"
    #
    ### N‑gram model hyperparameters
    "ngram_n": 3,  # n in n-gram
    "ngram_smoothing": True,  # Laplace smoothing
    "laplace_alpha": 1.0,  # α for Laplace smoothing
    "use_backoff": True,
    #
    ### Neural model training parameters
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 5,
    "max_length": 32,
    "dropout": 0.1,
    #
    ### LSTM model hyperparameters
    "embed_size": 256,
    "hidden_size": 512,
    "num_layers": 2,
    #
    ### Transformer fine‑tuning
    "pretrained_model_name": "t5-small",  # or t5-large, etc.
    "warmup_steps": 100,
    #
    ### Results
    "out_path": "out/",  # where to dump experiment outputs
    "save_metrics": ["bleu", "chrf"],
    #
    ### Random seed
    "seed": 42,
}

if __name__ == "__main__":
    # Quick test to print configuration values
    for key, value in CONFIG.items():
        print(f"{key}: {value}")
