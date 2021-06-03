# Prenyltransferase's Research on the Hyperparameters of Transformer

In this page we implement a Transformer model from the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper.

## Dataset

* A small-scale corpus of simple English sentences and Chinese translations:
  * Boil the water.,把水烧开。
  * Do you know where i live?,你知道我住哪吗？
  * What else have you figured out?,你还发现什么了？
* Unique tokens:
  * In Engllish: 3391; 
  * In Chinese: 4166.
* Number of examples:
  * 17587 training examples;
  * 2197 Validation examples;
  * 2189 testing examples.

## Something Unchanging

* Do not reverse the source (English) sentence.
* Use **a learned positional encoding** instead of a sinusoidal one.
* Do not change the internal structure of **multi-head attention layer**, except for the number of heads.
* Use the standard Adam optimizer with a static learning rate 0.0005 instead of one with warm-up and cool-down steps.
* Use **Xavier uniform** as a weight initialization scheme.
* Train on a single Tesla-K80, 256 for each batch, 30 epochs.

## Explanations of Hyperparameters

* HID_DIM: $d_{model}$ in the paper. All sub-layers in this model, as well as the embedding layers, produce outputs of dimension = HID_DIM.
* ENC_LAYERS/DEC_LAYERS: The encoder/decoder is composed of a stack of N = ENC_LAYERS/DEC_LAYERS identical layers. 
* N_HEADS: The number of heads in encoder's/decoder's multi-head attention layer.
* PF: Whether the position-wise feed-forward network exists in this model.
* PF_ACT: If the position-wise feed-forward network exists, its activation function.
* ENC_PF_DIM/DEC_PF_DIM: The dimension of position-wise feed-forward network.
* TE_TYPE: Without explanation, the paper says: *In the embedding layers, we multiply those weights by $\sqrt{d_{model}}$*. So TE_TYPE is used to decide whether to multiply this scaling factor.
* ENC_DROPOUT/DEC_DROPOUT: Apply dropout to the output of each sub-layer, sums of the embeddings and the positional encodings in both the encoder and decoder stacks. 

