<h2>Char-Transformer: Cute Character-Level Text Generator</h2>

A small transformer-based model for generating text at the character level. It uses cool tricks like rotary position encodings (ROPE) and multi-head attention to create interesting text from your input.
Features

Character-level generation: Works with individual characters, not words.

Custom DecoderBlock: Built with ROPE and multi-head attention for smarter text generation.

Easy training: Tracks loss, saves checkpoints, and trains on your text.

Requirements

        pip install torch tqdm pandas


<h2>Files</h2>

trainer.py: Handles the training loop and saves models.

utils.py: Includes a function to get sample text data.

tokenizer.py: Tokenizer class for turning text into characters and building vocab.

dataloader.py: Prepares your text data for training.

blocks.py: Contains the transformer layers, including ROPE and the decoder block.

train.py: The trainer.

<h2>How to Use</h2>

Provide your text data.

Run the training loop (it'll save the model after each epoch).

Enjoy your trained model generating cool character-based text!

<h2>TO-DO:</h2>

- [ ] Upload Weights

- [ ] Add pre-training ?

- [ ] Make the generator Work ?
