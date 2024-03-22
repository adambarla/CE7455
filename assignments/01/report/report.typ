#import "template.typ": *

#show: project.with(
  title: "Deep Learning models for Sentence Classification,
  Assingment 1, course CE7455",
  authors: (
    (name: "Adam Barla", email: "n2308836j@e.ntu.edu.sg", affiliation: "NTU, Singapore"),
  )
)

= Configuration Optimization

== <section_1a>
#text(fill:gray)[Implement the `pack_padded_sequence` function in PyTorch's RNN library. Report results under the default setting and discuss the benefits of this function.]

The function `pack_padded_sequence` takes input sequences of varying lengths and packs them into a compact form, before they are fed into the RNN.  By using pack_padded_sequence, the model can skip the padded areas, focusing only on the actual data, which can lead to more accurate and faster training.

I created a class `PackedRNN` that used this function.
Then, I conducted $10$ training runs of $100$ epochs each for model using the function and for the baseline model.
I compared the models on validation accuracy so the test accuracy can still serve as an indicator of the performace on unseen data in the end.
Average valildation accuracy of the baseline model was $67.17$% while model with `pack_padded_sequence` reached $68.53$%, which is a minor improvement.
The average duration of one epoch went up by ~$\~3.5$ seconds from $17.5$s to $21$s.
This can be improved by packing the sequences only once and not every time `forward` method is called.

==
#text(fill:gray)[Experiment with different configurations (optimizers, learning rates, batch sizes, sizes of hidden embedding) and report the best configuration's performance on the validation and test sets.]

For this purpouse I decomposed the notebook to a project.
Using `hydra` and `wandb` I created a #link("https://wandb.ai/crutch/Deep%20Learning%20models%20for%20Sentence%20Classification/sweeps/ppcppf7r?nw=nwusercrutch")[sweep] over hyperparameters.
I chose to test using grid search over these parameters based on some empirical test runs:

```yaml
  lr:
    values: [ 0.00001, 0.0001, 0.001, 0.01, 0.1 ]
  batch_size:
    values: [ 32, 64, 128 ]
  optimizer:
    values: [ sgd, adam, adadelta, adagrad, rmsprop ]
  model.hidden_dim:
    values: [ 50, 100, 200, 400 ]
```

I tested each combination of parameters with `PackedRNN` class from @section_1a for $100$ epochs. The best performance on validation accuracy was achieved when using these parameters:

```yaml
  hidden_dim: 200
  optimizer: adadelta
  batch_size: 32
  lr: 0.1
```

The validation and test accuracy reached was $77.71$% and $83.8$% respectively.
=== Parameter interaction analysis

@hparam_interaction shows an average validation accuracy when two parameters are kept at a specific value.
We can observe trends that help us better understand which hyperparameters are optimal.
For example we can see general correlation between hidden embedding size and validation accuracy, which is not suprising.
Adam and RMSprop optimizers seem to generally dominate, even though Adadelta reached the highest validation accuracy.  Each optimizer has a learning rate at which it works best as it seems.
Trend towards smaller batch sizes seems curious. Given more time I would explore it further.

#figure(
  image("./figures/hparam_interaction.png" , width: 90%),
  caption: [Each square represents a mean validation accuracy of runs with two parameters set to particular value.],
  placement: top,
) <hparam_interaction>

For the next experiments I will use Adam with hidden dimension of $200$, batch size of 32.
Learning rate may vary in the future but a good value for this configuration seems $0.001$.

==
#text(fill:gray)[Implement regularization techniques, describe them, and report accuracy results after application.
]

=== Dropout
During training, randomly zeroes some of the elements of the input tensor with probability $p$ given by a parameter, which greatly reduces overfitting.
This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons as described in the #cite("dropout").
Dropout has an effect of training and using an ensamble of models and promotes learning of a sparse representation.


=== L1/2 Regularization

We can include a regularization parameter to the loss, which is computed based on the parameters of the model.\
For L1 it is
$ alpha sum_(omega in Omega)|omega| $
and for L2 it is
$ alpha sum_(omega in Omega)omega^2 $
where $Omega$ is a set of all parameters of the model and $alpha$ is a weight of the regularization parameter.

This regularization results in sparsity.

=== Gradient clipping

To counteract exploding gradient we can employ gradient clipping.
It scales the gradient $bold(g)$ down if it's norm $norm(bold(g))$ is larger then some treshold $t$.

$ text("if") norm(bold(g)) >= t: bold(g) <- t * (bold(g)/norm(bold(g))) $

For this I used function `torch.nn.utils.clip_grad_norm`.

=== Early stoping
Up to a point, training improves the learner's performance on the validation set. Past that point, however, improving on the training data comes at the expense of increased generalization error.

I implemented early stoping with patience. If validation accuracy doesn't improve in `patience` epochs, the run is terminated. By using patience we can increase number of epochs to a large value because models stop by themselves.

=== Batch normalization

Batch normalization makes training of NNs faster and more stable through normalization of the layers' inputs by re-centering and re-scaling.
It can mitigate the problem of internal covariate shift, where parameter initialization and changes in the distribution of the inputs of each layer affect the learning rate of the network.

It isnt' common to use batch norm with RNNs but it can be used later with more complex classifiers in later sections.

=== Results of regularization <regularization>

I conducting another #link("https://wandb.ai/crutch/Deep%20Learning%20models%20for%20Sentence%20Classification/sweeps/nf1o9d3o?nw=nwusercrutch")[sweep] over the new hyperparameters.
I kept the previously found parameters and I additionaly set `patience` to $10$, number of epochs to $500$ and regularization to l2. I picked l2 over l1 by conducting smaller sweeps, where l2 always outperformed l1. I searched over these parameters:

```yaml
    lr:
        min: 0.00001
        max: 0.001
    dropout:
        min: 0.0
        max: 0.5
    regularizer.alpha:
        min: 0.00001
        max: 0.01
    grad_clip_threshold:
        min: 1.0
        max: 10.0
```

I searched over learning rate because regularization techniques might result in a different optimal learning rate.
The best parameters found were:
```yaml
  regularizer.alpha: 0.0006332825849228081
  lr: 0.0008726644867850513
  grad_clip_threshold: 3.409241906075233
  dropout: 0.014878368933965491
```
and the model with these parameters achieved validation and test accuracy reached was $86.88$% and $89.2$% respectively, which is an improvement.

= Input Embedding <gensim>
#text(gray)[Switch from randomly initialized input word embeddings to pre-trained word2vec embeddings. Report accuracy on the validation set and compare performance.
- You can refer to #link("https://radimrehurek.com/gensim/intro.html#installation")[here] on how to install gensim to in order to work with word2vec. @rehurek_lrec
- Pretrained word2vec models can be downloaded following #link("https://radimrehurek.com/gensim/models/word2vec.html#pretrained-model")[link]. Use `word2vec-google-news-300` as the pretrained word2vec embeddings.
]

I created a new class `GensimPackedRNN`. The `nn.Embedding` in this class is initialized through function `from_pretrained`.
This function requires an matrix of $text("vocab_size")times text("embedding_size")$. I create it for the vocabulary used before by iterating through the words and finding a corresponding embedding in the `word2vec-google-news-300`. For words that are not in the vocabulary of the word2vec model I just insert an empty vector as an embedding. In total, 419 out of 7687 weren't found. These include for example `<unk>` and `<pad>` tokens, symbols such as `?` and `-`, numbers, names and complex words such as `hendecasyllabic`.

The model, with same hyperparameters as in @regularization, #link("https://wandb.ai/crutch/Deep%20Learning%20models%20for%20Sentence%20Classification/runs/j6t300g6?nw=nwusercrutch")[reached] accuracy of $83.76$% and $86.6$% on validation and test data, which is a slight decrease that may have been caused by the missing embeddings.

= Output Embedding
#text(gray)[Explore options for computing sentence embedding beyond the final hidden representation. Implement the best option(s) and report accuracy on the validation set, comparing it to the performance in Task 2.]

== Average of Word Embeddings

One of the simplest methods is to take the average of the word embeddings of all words in the sentence. This method is easy to implement but does not take into account the order of the words in the sentence.

== Concatenation of First and Last Hidden States
In a bidirectional RNN, the first hidden state captures the "beginning" of the sentence, and the last hidden state captures the "end" of the sentence. Concatenating these two hidden states can provide a more comprehensive representation of the sentence.

== Max/Average Pooling of hidden states
Max or average pooling operations can be applied to the hidden states of an RNN to create a sentence embedding. Max pooling captures the most important features, while average pooling captures the average features of the hidden states.

== Self-Attention

To calculate the sentence embedding we can compute a weighted sum of all hidden states in the sentence. The weights are learned through self-attention during training and determine the importance of each word to the meaning of the sentence.

I opted for using this technique. I implemented `AttentionGensimPackedRNN` class that uses multiheaded attention:
```python
def forward(self, text, text_lengths):
  ...
  packed_output, _ = self.rnn(packed_input)
  padded_output, _ = pad_packed_sequence(packed_output)
  attention_output, _ = self.attention(
    padded_output, padded_output, padded_output
  )
  sentence_embedding = torch.mean(
    attention_output, dim=0
  )
  ...
```

Maintainig hyperparameters from before and using only $1$ attention head the model #link("https://wandb.ai/crutch/Deep%20Learning%20models%20for%20Sentence%20Classification/runs/j6t300g6?nw=nwusercrutch")[reached] accuracy of $85.32$% and $87.2$% on validation and test data, which is a slight increase compared to model form @gensim, but it still worse than model from @regularization. This could be possibly improved with further hyperparameter tuning.


= Architecture Optimization
#text(gray)[The original code base uses the most simple RNN architecture. Now consider more complex architectures in the following. Run experiments by replacing the original RNN with each of the following architectures. Report the accuracy on the validation set and compare these results. Discuss your findings.]

=== GRU with a single hidden layer
=== LSTM with a single hidden layer
=== Bidirectional simple RNN with a single hidden layer (consisting of a forward pass and a backward pass)
=== Simple RNN with 2 hidden layers]


= Critical Thinking
Propose and implement a modification to further improve performance. Conduct experiments and report accuracy on the validation set.
  

#counter(heading).update(0)

#bibliography("works.bib", style: "ieee", title: "References")