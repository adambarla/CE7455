#import "template.typ": *
#show: ieee.with(
  title: "Seq2Seq model for machine translation",
  // abstract: [],
  authors: (
    (
      name: "Adam Barla",
      department: [CE7455: Assignment 2],
      // organization: [Affiliation],
      location: [NTU, Singapore],
      email: "n2308836j@e.ntu.edu.sg"
    ),
  ),
  // index-terms: ("A", "B", "C", "D"),
  // bibliography-file: "refs.bib",
)

#show link: set text(blue)
#show link: underline

This project focused on getting familiar with different architectures of Sequence to Sequence model by integrating them to a prepared code base.

= Code Base
I started by rewriting the codebase to make it more modular and easier to understand. The code can be found attached in `02.zip` or in the #link("https://github.com/adambarla/ntu_nlp/tree/main/assignments/02")[repository].
Amongst other things, I
- created a `seq2seq` model class and separated backpropagation from forward pass
- adjusted the encoder forward pass to handle the whole sequence at once
- created a dataset and dataloader
- added support for batch processing
- added early stopping
- used #link("https://hydra.cc/docs/intro/")[hydra library] to easily change the configuration
- used #link("https://docs.wandb.ai/guides")[wandb] to log the training process

Batch size was set to $1$ to be consistent with the original code base. However, the code is prepared to handle batch processing.

= Methodology
I experimented with different configurations of the model and evaluated their performance on the test set with #link("https://en.wikipedia.org/wiki/ROUGE_(metric)")[ROUGE metric]. Results are shown in the @test_fmeasure, @test_recall, and @test_precision in the @results. All runs can be found on #link("https://wandb.ai/crutch/Seq2Seq%20model%20for%20machine%20translation?nw=45ooxwe4nje")[wandb].

Early stopping was used with patience of $3$ epochs.
Metric used for the early stopping was the validation loss.

Criterion used for training was the `torch.nn.NLLLoss` which required the use of `torch.nn.functional.log_softmax` instead of `torch.nn.functional.softmax`.
Optimizer chosen was the basic `torch.optim.SGD` optimizer with the default learning rate of $0.1$.

During the forward pass of the model, teacher forcing was used with a probability of $0.5$.
Teacher forcing is a technique used in training recurrent neural networks that speeds up convergence and improves the model's performance by feeding the true target sequence as input to the decoder at each time step.

 == Data
Data used for training can be found #link("https://www.manythings.org/anki/")[here].
The dataset consisted of $232736$ pairs of English and French sentences.
The maximal length of the input and output sequences was set to $15$.
Longer sequences were not used, so the final dataset consisted of only $22907$ sentences.
It was split into train, validation, and test sets with a ratio of $80\%$, $10\%$, and $10\%$ respectively.
 All models were trained with word tokenizer and the same vocabulary size.
There were $7019$ words in the  english tokenizer and $4638$ in the french one.

== Metrics
=== Rouge 1
Measures the overlap of 1-gram (each word) between the machine-generated text and the reference text. It focuses on the extraction of key terms and is a measure of content overlap.

=== Rouge 2
Measures the overlap of bigrams (two consecutive words) between the machine-generated text and the reference text. It gives insight into the phrase-level accuracy of the model.

=== Rouge L
Measures the longest matching sequence of words using longest common subsequence (LCS) statistics, which can capture sentence-level structure similarity.
It is less sensitive to sentence variations and often used to assess the fluency of translations.

=== Rouge L-Sum
Combines the ROUGE-L score for each sentence in the summary or translation and then sums these scores.
It is a cumulative measure of the longest common subsequences across the entire document rather than for individual sentences.
This metric is particularly useful for evaluating longer documents where the structure and flow of information are important, as it captures the overall fluency and coherence across multiple sentences.
High ROUGE-L-Sum scores suggest that the model is effective at translating longer passages with coherent and consistent sentence structures.

== Experiments

=== Baseline (GRU)

I started with the baseline model, which is a simple GRU (`torch.nn.GRU`) encoder-decoder model.

The run command for the baseline model is:
```bash
python -m train
```
and the default parameters defined in `src/conf/main.yaml` are used.

Run statistics and log can be seen #link("https://wandb.ai/crutch/Seq2Seq%20model%20for%20machine%20translation/runs/fg4m6cp8?nw=nwusercrutch")[here].

=== LSTM
Second, I replaced GRU with LSTM (`torch.nn.LSTM`) in both encoder and decoder components by adjusting the the run command as so:
```bash
python -m train encoder=lstm decoder=lstm
```

Run statistics and log can be seen #link("https://wandb.ai/crutch/Seq2Seq%20model%20for%20machine%20translation/runs/ucwz36tp?nw=nwusercrutch")[here].


=== Bidirectional LSTM
I then changed the encoder LSTM to bidirectional. This required another adjustment to the decoder to handle the bidirectional output of the encoder. Hidden state of the decoder was initialized with the concatenation of the forward and backward hidden states of the encoder. Therefore the hidden size of the decoder was doubled.

The run command for the bidirectional LSTM model is:
```bash
python -m train encoder=lstm decoder=lstm \
 encoder.bidirectional=True
```
I chose to keep the decoder as LSTM as it wasn't clear if I should use the GRU. In other configurations, it was explicitly stated to use the original code base. However, to use gru in the decoder just change the `decoder` parameter to `gru` in the command above or remove it as it is the default.

Run statistics and log can be seen #link("https://wandb.ai/crutch/Seq2Seq%20model%20for%20machine%20translation/runs/f33mny1s?nw=nwusercrutch")[here].

=== Attention Mechanism
I integrated an attention mechanism between the encoder and decoder.

Following code is a part of the decoder forward pass that applies the attention to create new embedding for input.
Embedding `emb` is of shape `1 x B x H`, where `B` is the batch size and `H` is the hidden size. `e_out` is the output of the encoder, of shape `L x B x H`, where `L` is the length of the input sequence.

```python
if self.use_attention:
    e_out_T = e_out.permute(1, 0, 2) # B x L x H
    emb_T = emb.permute(1, 2, 0) # B x H x 1
    att_s = torch.bmm(e_out_T, emb_T) # B x L x 1
    att_w = F.softmax(att_s, dim=1)
    att_w = att_w.transpose(1, 2) # B x 1 x L
    # context vector (weighted sum of encoder outputs)
    emb = torch.matmul(att_w, e_out_T) # B x 1 x H
    emb = emb.reshape(1, B, -1) # 1 x B x H
```

This code could be improved by passing multiple inputs of the decoder at once, which would allow for parallel computation of the attention mechanism. However, this is not possible due to teacher forcing.

GRU was used as the encoder and decoder.
The command used for the attention mechanism model is:
```bash
python -m train decoder.use_attention=true
```

Run statistics and log can be seen #link("https://wandb.ai/crutch/Seq2Seq%20model%20for%20machine%20translation/runs/e25rj2zg?nw=nwusercrutch")[here].


=== Transformer Encoder
I created a `TransformerEncoder` class that uses `torch.nn.TransformerEncoder`.
The decoder remained GRU.

The following code is the forward pass of the encoder. The input `x` is of shape `L x B`, where `L` is the length of the input sequence and `B` is the batch size. The input to the decoder is created by taking averaging output of the encoder over the sequence length. Mask is created to mask the padding tokens. Positional encoding is added to the input embeddings. The implementation of the positional encoding was taken from #link("https://pytorch.org/tutorials/beginner/transformer_tutorial.html")[here].

```python
def forward(self, x): # x: L x B
    src = self.embedding(x)
    src *= math.sqrt(self.hidden_size)
    src = self.pos_encoder(src)
    src_mask = self._get_mask(x)
    output = self.TransformerEncoder(src,
                    src_key_padding_mask=src_mask)
    return output, output.mean(dim=0).unsqueeze(0)
```

Following parameters of the transformer were set to defaults from #link("https://arxiv.org/abs/1706.03762")[Attention is All You Need paper]. Hidden size was the same in the paper as in the other experiments.
```
dim_feedforward: 2048
nhead: 8
num_layers: 6
d_model: 512
dropout: 0.1
```

Batch size was set to $10$ this time as it resulted in better performance. The command used for the transformer encoder model is:
```bash
python -m train batch_size=10 encoder=transformer
```

Run statistics and log can be seen #link("https://wandb.ai/crutch/Seq2Seq%20model%20for%20machine%20translation/runs/dxjx0b8a?nw=nwusercrutch")[here].

= Results <results>

The experiments evaluated the performance of various configurations of the Seq2Seq model on machine translation tasks. The configurations included GRU, GRU with attention, LSTM, bidirectional LSTM, and transformer encoder models. The models were assessed using the Rouge F-Measure and Precision metrics across 1-, 2-, and L-grams.

Results showed that the integration of attention mechanisms generally improved performance. GRU with attention consistently outperformed all the other configurations in all metrics.

The bidirectional LSTM outperformed the regular LSTM configuration on all metrics but it was worse than any GRU model which may attributed to the complexity of LSTM and challenges that come with training t. This result suggests that bidirectional processing of input enhanced the model translations capabilities.

The model with the Transformer encoder outperformed the LSTM models in all metrics the base GRU in `rouge 1` `rouge 2` and `rouge L-Sum` but it didn't reach the performance of the GRU model with attention in any of the metrics.
During training it reached the lowes training and validation loss, but it didn't translate to better performance on the test set.

Overall, attention (present in Transformer too) showed the best balance between recall and precision, indicating its potential as a robust approach for machine translation tasks.

#figure(
    [
        #show table.cell.where(x: 10): set text(
          weight: "bold",
        )
        #let t_csv = csv("figures/test_fmeasure.csv")
        #let t = t_csv.map(m => {
        // if v contains * then bold
        m.map(v => {
            if v.contains("*") {
                return [*#v.replace("*", "")*]
            } else {
                return v
            }
        })
        })
        #pad(y: 10pt, x: 5pt,
            table(
                stroke:  none,
                columns: (85pt,) +  (t.first().len()-1) * (1fr,),
                align: (horizon, center),
                table.hline(start: 0,stroke:1pt),
                table.header(
                table.cell(rowspan:2,[*Configuration*]), table.cell(colspan: 4, [*Rouge F-Measure*]),
                table.hline(start: 0,stroke:0.5pt),
                [1], [2], [L], [L-Sum],
                ),
                table.hline(start: 0),
                //table.vline(x: 1, start: 1),
                //table.vline(x: 10, start: 1),
                ..t.flatten(),
                table.hline(start: 0,stroke:1pt),
            )
        )
    ],
    caption: [Test F-Measures of the different configurations of the model. The best results are highlighted in bold.],
) <test_fmeasure>


#figure(
    [
        #show table.cell.where(x: 10): set text(
          weight: "bold",
        )
        #let t_csv = csv("figures/test_recall.csv")
        #let t = t_csv.map(m => {
        // if v contains * then bold
        m.map(v => {
            if v.contains("*") {
                return [*#v.replace("*", "")*]
            } else {
                return v
            }
        })
        })
        #pad(y: 10pt, x: 5pt,
            table(
                stroke:  none,
                columns: (85pt,) +  (t.first().len()-1) * (1fr,),
                align: (horizon, center),
                table.hline(start: 0,stroke:1pt),
                table.header(
                table.cell(rowspan:2,[*Configuration*]), table.cell(colspan: 4, [*Rouge Recall*]),
                table.hline(start: 0,stroke:0.5pt),
                [1], [2], [L], [L-Sum],
                ),
                table.hline(start: 0),
                //table.vline(x: 1, start: 1),
                //table.vline(x: 10, start: 1),
                ..t.flatten(),
                table.hline(start: 0,stroke:1pt),
            )
        )
    ],
    caption: [Test Recall of the different configurations of the model. The best results are highlighted in bold.],
) <test_recall>

#figure(
    [
        #show table.cell.where(x: 10): set text(
          weight: "bold",
        )
        #let t_csv = csv("figures/test_precision.csv")
        #let t = t_csv.map(m => {
        // if v contains * then bold
        m.map(v => {
            if v.contains("*") {
                return [*#v.replace("*", "")*]
            } else {
                return v
            }
        })
        })
        #pad(y: 10pt, x: 5pt,
            table(
                stroke:  none,
                columns: (85pt,) +  (t.first().len()-1) * (1fr,),
                align: (horizon, center),
                table.hline(start: 0,stroke:1pt),
                table.header(
                table.cell(rowspan:2,[*Configuration*]), table.cell(colspan: 4, [*Rouge Precision*]),
                table.hline(start: 0,stroke:0.5pt),
                [1], [2], [L], [L-Sum],
                ),
                table.hline(start: 0),
                //table.vline(x: 1, start: 1),
                //table.vline(x: 10, start: 1),
                ..t.flatten(),
                table.hline(start: 0,stroke:1pt),
            )
        )
    ],
    caption: [Test Precision of the different configurations of the model. The best results are highlighted in bold.],
) <test_precision>