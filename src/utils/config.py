CONFIG = {
    ### Data paths
    "data_path": "data/processed/",  # Path to your preprocessed dataset
    "raw_data_path": "data/raw/",  # Path to raw data
    "dataset": "shakez",  # "shakez", "sonnetz"
    #
    ### N‑gram model hyperparameters
    "ngram_n": 4,  # n in n-gram (e.g., 5 for 5‑gram)
    "ngram_smoothing": True,  # Whether to apply Laplace smoothing
    "laplace_alpha": 1.0,  # α for Laplace (add‑α) smoothing
    "use_backoff": True,  # Whether to back off to unigram when unseen
    #
    ### Neural model training parameters
    "batch_size": 32,
    "learning_rate": 1e-5,
    "num_epochs": 3,
    "max_length": 32,
    "dropout": 0.1,
    #
    ### LSTM model hyperparameters
    "tokenizer": "bert-base-uncased",
    "embed_size": 256,
    "hidden_size": 512,
    "num_layers": 2,
    #
    ### Transformer fine‑tuning
    "pretrained_model_name": "t5-small",  # or t5-large, etc.
    "warmup_steps": 100,  # for transformer scheduler
    #
    ### Results
    "out_path": "out/",  # where to dump experiment outputs
    "save_metrics": ["bleu", "chrf"],  # always save these metrics in save_results()
    #
    ### Random seed
    "seed": 42,
}

if __name__ == "__main__":
    # Quick test to print configuration values
    for key, value in CONFIG.items():
        print(f"{key}: {value}")
