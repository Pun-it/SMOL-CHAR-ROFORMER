<h1>SMOL-CHAR-ROFORMER: Cute Character-Level Text Generator with RoPE</h1>

Rope-mechanism is implimented from scratch using torch. 
 
    @article{su2021roformer,
    title={Roformer: Enhanced Transformer with Rotary Position Embedding},
    author={Mingxing Tan and Ruoming Pang and Yi Yang and Wei Wu and Nan Du},
    journal={arXiv preprint arXiv:2104.09864},
    year={2021},
    url={https://arxiv.org/abs/2104.09864}
    }


A small transformer-based model for generating text at the character level. It uses cool tricks like rotary position encodings (ROPE) and multi-head attention to create interesting text from your input.

<h2>Features</h2>

Character-level generation: Works with individual characters, not words.

Custom DecoderBlock: Built with ROPE and multi-head attention for smarter text generation.

Easy training: Tracks loss, saves checkpoints, and trains on your text.

Requirements

    
    pip install torch tqdm pandas


<h2>Files</h2>

  trainer.py: Handles the training loop and saves models.
  
  train.py: Defines the model and trains it.
  
  utils.py: Includes a function to get sample text data.
  
  tokenizer.py: Tokenizer class for turning text into characters and building vocab.
  
  dataloader.py: Prepares your text data for training.
  
  blocks.py: Contains the transformer layers, including ROPE and the decoder block.
  
  generator.py: Contains the code to generate text after training.
  
  generate.py: Does not work yet, will do, soon enough.

<h2>How to Use</h2>

  Provide your text data.  
  
  Run the training loop (it'll save the model after each epoch).
  
  Enjoy your trained model generating cool character-based text!

TODO : 
- [ ] Add weights
- [ ] Make the generator work
- [ ] Post training ?
