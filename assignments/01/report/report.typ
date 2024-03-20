#import "template.typ": *


#show: project.with(
  title: "Deep Learning models for Sentence Classification,
  Assingment 1, course CE7455",
  authors: (
    (name: "Adam Barla", email: "n2308836j@e.ntu.edu.sg", affiliation: "NTU, Singapore"),
  )
)
#set heading(numbering: "1.a")
  
= Configuration Optimization

==
#text(fill:gray)[Implement the `pack_padded_sequence` function in PyTorch's RNN library. Report results under the default setting and discuss the benefits of this function.]

The function `pack_padded_sequence` takes input sequences of varying lengths and packs them into a compact form, before they are fed into the RNN.  By using pack_padded_sequence, the model can skip the padded areas, focusing only on the actual data, which can lead to more accurate and faster training.

I conducted $10$ training runs of $100$ epochs each for model using the function and for the baseline model. I compared the models on validation accuracy so the test accuracy can still serve as an indicator of the performace on unseen data in the end. Average valildation accuracy of the baseline model was $67.17$% while model with `pack_padded_sequence` reached $68.53$%, which is a minor improvement. The average duration of one epoch went up by $~3.5$ seconds from $17.5$s to $21$s. This can be improved by packing the sequences only once and not every time `forward` method is called.

==
#text(fill:gray)[Experiment with different configurations (optimizers, learning rates, batch sizes, sizes of hidden embedding) and report the best configuration's performance on the validation and test sets.]


==
#text(fill:gray)[Implement regularization techniques, describe them, and report accuracy results after application.
]



= Input Embedding
Switch from randomly initialized input word embeddings to pre-trained word2vec embeddings. Report accuracy on the validation set and compare performance.

Gensim installation and pretrained word2vec models: [Gensim](https://radimrehurek.com/gensim/intro.html#installation), [Pretrained models](https://radimrehurek.com/gensim/models/word2vec.html#pretrained-models).

= Output Embedding
Explore options for computing sentence embedding beyond the final hidden representation. Implement the best option(s) and report accuracy on the validation set, comparing it to the performance in Task 2.

= Architecture Optimization
Experiment with more complex RNN architectures (GRU, LSTM, Bidirectional simple RNN, simple RNN with 2 hidden layers) and report accuracy on the validation set.

= Critical Thinking
Propose and implement a modification to further improve performance. Conduct experiments and report accuracy on the validation set.
  

#counter(heading).update(0)

#bibliography("works.bib", style: "ieee", title: "References")