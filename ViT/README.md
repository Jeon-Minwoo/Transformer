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

## Vision Transformers
### Class Token
Like BERT([arXiv](https://arxiv.org/abs/1810.04805), [PDF](https://arxiv.org/pdf/1810.04805.pdf)), transformer trains class token by passing it through multiple encoder blocks. The class token first initialized with zeros and appended to the input. Just like BERT, the NLP transformer also uses class token. Consequently, class token was inherited to vision transformer too. On vision transformer. The class token is a special symbol to train. Even if it looks like a part of the input, as long as it's a trainable parameter, it make more sense to treat it as a part of the model. 

### Positional Embedding
First, you should understand that sequence is a kind of position. The authors of NLP transformer tried to embed fixed positional values to the input and the value was formulated as ${p_t}^{(i)} := \begin{cases} \sin(w_k \bullet t) \quad \mathrm{if} i=2k \\ \cos(w_k \bullet t) \quad \mathrm{if} i=2k+1 \\ \end{cases}, w_t = {1 \over {10000^{2k/d}}}$. This represents unique positional information to all tokens. On the other hand, vision transformer, set the positional information as another learnable parameter. After the training, the positional vector is looks like Figure5.

> <img src='./archive/img/01. preceding works/01. attention mechanism/05. positional embedding.png' />
> Figure5. Position embeddings of models trained with different hyperparameters.

### GELU Activation
They applied GELU activation function([arXiv](https://arxiv.org/abs/1606.08415), [PDF](https://arxiv.org/pdf/1606.08415.pdf)) proposed by Dan Hendrycks and Kevin Gimpel. They combined dropout, zoneout and ReLU activation function to formulate GELU. ReLU gives non-linearity by dropping negative outputs and os as GELU. Let $x\Phi(x) = \Phi(x) \times Ix + (1 - \Phi(x)) \times 0x$, then $x\Phi(x)$ defines decision boundary. Refer to the paper, loosely, this expression states that we scale $x$ by how much greater it is than other inputs. Since, the CDF of a Gaussian is often computed with the error function, they defiend Gaussian Error Linear Unit (GELU) as $\textrm{GELU}(x) = xP(X \le x) = x\Phi(x)=x\bullet {1 \over 2}[\textrm{erf}({x \over \sqrt{2}})]$. and we can approximate this with $\mathrm{GELU}(x) = 0.5x(1+\tanh[\sqrt{2 \over \pi}(x + 0.044715x^3)])$.

> <img src='./archive/img/01. preceding works/01. attention mechanism/03. gelu.png' /> <br />
> Figure3. The $\mathrm{GELU} (\mu=0,\sigma=1)$, $\mathrm{ReLU}$ and $\mathrm{ELU} (\alpha=1)$.
> 
> <img src='archive/img/01. preceding works/01. attention mechanism/04. gelu_performance.png' /> <br />
> Figure4. MNIST Classification Results. Left are the loss curves without dropout, and right are curves with a dropout rate of 0.5. Each curve is the the median of five runs. Training set log losses are the darker, lower curves, and the fainter, upper curves are the validation set log loss curves.

See the [paper](https://arxiv.org/abs/1606.08415) for more experiments.

## Experiments
### Settings & Environments
Because our computing resource is limited, it wasn't possible to follow exactly same batch size and traning procedure.
