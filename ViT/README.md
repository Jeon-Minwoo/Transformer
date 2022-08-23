# Vision Transformer
Implementation of 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale' ICLR 2020 ([arXiv](https://arxiv.org/abs/2010.11929), [PDF](https://arxiv.org/pdf/2010.11929)) 
and a documentation about its historical and technical background.
Some figures and equations are from their original paper.
![vision transformer](./archive/img/01.%20vision%20transformer.png)

## Index
> 1. Preceding Works
>    * Attention Mechanism
>    * Transformers 
> 2. Vision Transformers
>    * Overall Structure
>    * Patching
>    * Positional Embedding
>    * Layer Normalization
>    * GELU Activation
> 3. Experiments
>    * Environments
>    * Result

## Preceding Works
### Attention Mechanism
#### Attention in NLP
In the field of NLP, they pushed the limitations of RNNs by developing [attention mechanism](https://arxiv.org/abs/1409.0473). Attention mechanism is a method to literally pay attention to the important features. In NLP, for example, with the sentence "I love NLP because it is fascinating.", the word 'it' referes to 'NLP'. Thus, when you process 'it', you have to treat it as 'NLP'. 
> <img src='./archive/img/01. preceding works/01. attention mechanism/01. attention score_nlp.png' />
> <p>Figure1. An example of attention score between words in a sentence. You can see 'it' and 'NLP' has high attention score. This examples represents one of attention mechanisms called self-attention.</p>

Attention is basically inner product operation as similarity measurement. You have three following informations:
|Acronym|Name|Description|En2Ko example|
|:-:|-|-|-|
|Q|Query|An information to compare this with all keys to find the best-matching key.|because|
|K|Key|A set of key that leads to appropriate values.|I, love, NLP, because, it, is, fascinating|
|V|Value|The result of attention.|나, 자연어처리, 좋, 왜냐하면, 이것, 은, 흥미롭다|
When you translate "because" in "I love NLP because it is fascinating.", first you calculate the similarity between "because" and other words. Then, weight-sum the korean words with the similarity, you get an appropriate word vector.

To take attention mechanism further, Ashish Vaswani et al. introduced a framework named called "[Transformer](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)". Transformer takes a sequential sentence as a pile of words and extracts features through attention encoder and decoders with several types of attention. More details of transformer structure is explained on the "Transformer" section. While attention mechanism sounds quite reasonable with NLP examples, it's not trivial that it works in computer vision as well. Alexey Dosovitskiy from Google Research introduced transformer for computer vision and explained how it works.

#### Attention in Computer Vision
By attention mechanism, in NLP, the model calculates where to concentrate. On the other hand, in computer vision, Jie Hu et al. proposed a new attention structure for computer vision (Squeeze-and Excitation Networks, CVPR 2018, [arXiv](https://arxiv.org/abs/1709.01507), [PDF](https://arxiv.org/pdf/1709.01507.pdf)). The proposed architecture contains a sub-network that learns channel-wize attention score.

> <img src='./archive/img/01. preceding works/01. attention mechanism/02. se module.png' />
> <p>Figure2. This diagram shows the structure of squeeze-and-excitation module. A sub-network branch makes 1x1xC dimensional tensor of attention scores for the input of the module.</p>

Convolutional neural networks, is in a way, a channel-level ensemble. Each kernels extracts different features from an image for another features. But these features from different kernels are not always important equally. SE module helps emphasize the important features and reduce other features.

Another way to adapt attention is to split an image into fixed size patches and calculates similarity between all combinations of the patches. Detailed description of this method is in the "Transformer' section.

