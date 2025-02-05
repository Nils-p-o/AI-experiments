# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from training.experiment import TransformerExperiment
# from transformer_arch.components import ClassicTransformer
# from training.utils import dummy_dataloader, count_parameters

# if __name__ == "__main__":
#     # Model selection
#     model = ClassicTransformer(d_model=256, nhead=8, num_layers=4, d_ff=1024, dropout=0.1, vocab_size=1000, seq_len=128)

#     experiment = TransformerExperiment(model, learning_rate=5e-5, batch_size=16)

#     # Checkpointing
#     checkpoint_callback = ModelCheckpoint(
#         dirpath="checkpoints/",
#         filename="{epoch}-{val_loss:.2f}-{val_acc:.2f}",
#         save_top_k=3,  # Save the top 3 checkpoints based on validation loss
#         monitor="val_loss",  # Monitor validation loss for saving the best model
#         mode="min",  # Lower validation loss is better
#     )

#     # Early Stopping
#     early_stopping_callback = EarlyStopping(
#         monitor="val_loss", patience=3, verbose=True, mode="min"
#     )

#     trainer = pl.Trainer(
#         max_epochs=10,
#         accelerator="auto",
#         devices="auto",
#         callbacks=[checkpoint_callback, early_stopping_callback],
#     )
#     trainer.fit()


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from training.experiment import TransformerExperiment
from transformer_arch.components import ClassicTransformer 
from training.utils import count_parameters, ShakespeareDataModule, download_and_split_shakespeare


if __name__ == "__main__":
    # --- Data Loading ---
    download_and_split_shakespeare()  # Download and prepare data if needed

    seq_len = 128  # Or whatever sequence length you want
    batch_size = 32 # Choose an appropriate batch size

    data_module = ShakespeareDataModule(
        train_file='train.txt',
        val_file='val.txt',
        test_file='test.txt',
        seq_len=seq_len,
        batch_size=batch_size
    )
    data_module.setup() # Very important to setup the data
    vocab_size = data_module.get_vocab_size()

    # --- Model Definition ---
    model = ClassicTransformer(d_model=128, nhead=4, num_layers=3, d_ff=512, dropout=0.1, vocab_size=vocab_size, seq_len=seq_len)
    # Print parameter count:
    num_params = count_parameters(model)
    print(f"The model has {num_params:,} trainable parameters.")


    # --- Training Setup ---
    experiment = TransformerExperiment(model, learning_rate=2e-4, vocab_size=vocab_size)  # Use vocab_size

    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch}-{val_loss:.2f}-{val_perplexity:.2f}", # Use Perplexity, better metric
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    # Early Stopping
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=5, verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=10,  # Or use max_steps for finer control with LR scheduling
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback, early_stopping_callback],
        # gradient_clip_val=0.5,  # Optional gradient clipping
    )

    # --- Training ---
    trainer.fit(experiment, datamodule=data_module)  # Pass the data_module

    # --- Testing (Optional) ---
    trainer.test(datamodule=data_module)