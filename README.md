Char-Transformer: Cute Character-Level Text Generator

A small transformer-based model for generating text at the character level. It uses cool tricks like rotary position encodings (ROPE) and multi-head attention to create interesting text from your input.
Features

    Character-level generation: Works with individual characters, not words.
    Custom DecoderBlock: Built with ROPE and multi-head attention for smarter text generation.
    Easy training: Tracks loss, saves checkpoints, and trains on your text.

Requirements

bash```
pip install torch tqdm pandas
```

Files

    trainer.py: Handles the training loop and saves models.
    utils.py: Includes a function to get sample text data.
    tokenizer.py: Tokenizer class for turning text into characters and building vocab.
    dataloader.py: Prepares your text data for training.
    blocks.py: Contains the transformer layers, including ROPE and the decoder block.

How to Use

    Provide your text data.
    Run the training loop (it'll save the model after each epoch).
    Enjoy your trained model generating cool character-based text!