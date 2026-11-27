# Chapter 13. Sequence-to-Sequence and the Encoder–Decoder

> **Part 3 — Sequences and Attention** · Level: Intermediate
>
> **In this chapter you will learn:** how an encoder–decoder model maps one sequence to another of a different length · how such a model is trained with teacher forcing and how it generates output · why a single context vector becomes a bottleneck for long inputs

## From one sequence to another

How would you build a neural network that translates "I love you" into the Russian «Я тебя люблю» or the Korean "사랑해"? The English sentence has three words, the Russian three, the Korean one. The word order is different too. A language model from Chapter 11 produces one output per input token, so input and output are aligned position by position. Translation is not like that: the output has its own length and its own order.

Many problems share this shape. Summarization turns a long document into a short text. Question answering turns a question into an answer. Speech recognition turns a sequence of audio frames into a sequence of characters. Even converting a date written as "27 November 2026" into "2026-11-27" is a mapping from one sequence to another. Models for such tasks are called **sequence-to-sequence** models (seq2seq).

The solution proposed in 2014 — by Sutskever, Vinyals and Le, and in parallel by Cho and colleagues — is simple and elegant. Use two recurrent networks:

1. An **encoder** reads the whole input sentence and compresses it into a single vector, the **context vector** $c$.
2. A **decoder** starts from this vector and generates the output sentence one token at a time, like a language model that is "conditioned" on $c$.

This is the **encoder–decoder architecture**. It is one of the most important ideas in this book: the original Transformer (Chapter 19) and models such as T5 keep exactly this division into an encoder and a decoder, even though they replace the RNNs with attention.

## The encoder and the decoder

Let the source sentence be $x_1, \dots, x_S$ and the target sentence $y_1, \dots, y_T$. Note that $S$ and $T$ can be different.

The **encoder** is an RNN (in practice usually an LSTM or GRU from Chapter 12). It reads the source tokens one by one:

$$h_s = \operatorname{RNN}_{enc}(x_s, h_{s-1}), \quad s = 1, \dots, S$$

Its last hidden state becomes the context vector: $c = h_S$. If the encoder has done its job, $c$ contains everything needed to produce the translation.

The **decoder** is another RNN with its own weights. Its initial state is the context vector, $s_0 = c$. At each step it receives the previous output token and updates its state:

$$s_t = \operatorname{RNN}_{dec}(y_{t-1}, s_{t-1})$$

$$P(y_t \mid y_{<t}, x) = \operatorname{softmax}(W_{out}\, s_t + b_{out})$$

Here $y_{<t}$ means "all target tokens before position $t$", and $W_{out}$ maps the decoder state to one score (logit) per vocabulary word. In some versions, $c$ is also fed to the decoder at every step, not only at the beginning.

Two details are needed to make this work. First, the decoder needs something to read at step 1, before any word has been produced. We use a **special token** such as `<bos>` ("beginning of sequence") as $y_0$. Second, the decoder must be able to decide when the output is finished. We add an **end-of-sequence token** `<eos>` to every target sentence during training; at generation time we stop as soon as the decoder produces it.

Together, the decoder defines the probability of the whole output sentence by the chain rule of probability (Chapter 6):

$$P(y_1, \dots, y_T \mid x) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, x)$$

This is the same factorization as in an ordinary language model (Chapter 1), except that every factor also depends on the source sentence $x$. A seq2seq model is therefore a **conditional language model**.

Here is a toy encoder–decoder in NumPy with random (untrained) weights. It shows the data flow, not a working translator:

```python
import numpy as np

rng = np.random.default_rng(0)
V, d = 6, 8                       # vocabulary size, hidden size
BOS, EOS = 0, 1                   # special tokens: begin / end of sequence
E = rng.normal(0, 0.5, (V, d))    # embedding table (Chapter 5)
W_enc, U_enc = rng.normal(0, 0.5, (d, d)), rng.normal(0, 0.5, (d, d))
W_dec, U_dec = rng.normal(0, 0.5, (d, d)), rng.normal(0, 0.5, (d, d))
W_out = rng.normal(0, 0.5, (V, d))   # hidden state -> logits over vocabulary

def encode(src):
    h = np.zeros(d)
    for tok in src:                    # read the whole source sentence
        h = np.tanh(W_enc @ E[tok] + U_enc @ h)
    return h                           # context vector c = last hidden state

def greedy_decode(c, max_len=10):
    s, tok, out = c, BOS, []           # decoder starts from c and <bos>
    for _ in range(max_len):
        s = np.tanh(W_dec @ E[tok] + U_dec @ s)
        tok = int(np.argmax(W_out @ s))   # pick the most probable token
        if tok == EOS:
            break
        out.append(tok)                # feed our own prediction back in
    return out

c = encode([2, 3, 4, 5])
print("context vector:", np.round(c, 2))
print("generated ids:", greedy_decode(c))
```

Because the weights are random, the output is meaningless, and in this run the decoder never produces `<eos>`: it stops only because of `max_len`. This is a useful reminder that every generation loop needs a length limit.

## Training with teacher forcing

During training we have pairs of sentences (source, target), for example from a **parallel corpus** of translated texts. The loss is the cross-entropy (Chapter 8) of the correct target token at every step, summed over the sentence:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t \mid y_{<t}, x)$$

The gradient flows from the decoder back through $c$ into the encoder, so both networks are trained together, end to end.

An important question: what should the decoder receive as input at step $t$ during training — its own previous prediction, or the correct previous word? Early in training the model's predictions are mostly wrong. If we fed them back, one mistake would push the decoder into a strange state, and all later steps would learn from garbage. Instead, we feed the **correct** previous token $y_{t-1}$ from the training data, no matter what the model predicted. This is called **teacher forcing**: a teacher tells the student the right answer before every next question. It makes training stable and fast, and it lets all steps of the decoder be computed with known inputs.

Teacher forcing has a cost. At generation time there is no teacher; the decoder must consume its own predictions, as the loop in the code does. The model is therefore tested in a situation it never saw during training — after its own mistakes. This mismatch is known as **exposure bias**. Several remedies exist (for example, sometimes feeding the model's own predictions during training, a method called scheduled sampling), but teacher forcing remains the standard way to train both seq2seq models and the large language models of later chapters.

## Generating the output: greedy and beam search

At generation time we want the most probable output sentence:

$$\hat{y} = \arg\max_{y} P(y \mid x)$$

Trying every possible sentence is impossible: with a vocabulary of $V$ words there are $V^T$ sentences of length $T$. The simplest approximation is **greedy decoding**: at each step take the single most probable token, as `np.argmax` does in the code above. It is fast, but a choice that looks best now may lead to a poor sentence later.

**Beam search** keeps more options open. Instead of one partial sentence, it keeps the $k$ best ones (the "beam", with typical $k$ between 4 and 10). At each step it extends each of them with every possible next token, scores all candidates by their total log-probability $\sum_t \log P(y_t \mid y_{<t}, x)$, and keeps the best $k$. A small example with $k = 2$: if after step 1 the beam is {"Я" (log-prob $-0.4$), "Мне" ($-1.2$)}, then at step 2 we extend both and might keep {"Я тебя" ($-0.9$), "Я люблю" ($-1.5$)} — the second hypothesis from step 1 has been dropped. Because longer sentences collect more negative terms, beam search usually divides the score by the length to avoid preferring very short outputs. We return to decoding strategies, including sampling, in Chapter 24.

Sutskever et al. also found a curious trick: feeding the source sentence to the encoder **in reverse order** improved translation quality. With reversal, the first source words end up close to the first target words in time, which makes the early dependencies short and easy to learn. This trick is a hint of the real problem, discussed next.

## The bottleneck problem

The encoder must compress the entire source sentence — whether it has 5 words or 50 — into one vector $c$ of fixed size. For short sentences this works well. For long sentences it does not: Cho et al. (2014) observed that the translation quality of such models drops sharply as the source sentence becomes longer.

The reason is easy to understand. A vector of, say, 1,000 numbers must hold the meaning, the word order, the names and the grammatical details of an entire paragraph, and it must hold them in a form that the decoder can use at every step. The decoder, meanwhile, needs different information at different moments: when producing the verb it needs the source verb, when producing a name it needs the name. Yet it only sees the same compressed summary. This is the **information bottleneck** of the basic encoder–decoder.

Think of a human translator. They do not read a whole paragraph, close the book, and then write the translation from memory. They look back at the source text again and again, each time focusing on the part they are translating now. Chapter 14 shows how to give a neural network exactly this ability. The mechanism is called **attention**, and it changed the field.

## Summary

- Sequence-to-sequence models map an input sequence to an output sequence of a different length, as in translation or summarization.
- An encoder RNN reads the source and produces a context vector $c$; a decoder RNN generates the target from $c$ one token at a time.
- Special tokens `<bos>` and `<eos>` start the decoder and mark the end of the output; the model is a conditional language model $P(y \mid x) = \prod_t P(y_t \mid y_{<t}, x)$.
- Teacher forcing feeds the correct previous token during training; the mismatch with generation is called exposure bias.
- Greedy decoding takes the best token at each step; beam search keeps the $k$ best partial outputs.
- A single fixed-size context vector is a bottleneck, and translation quality drops for long sentences — the motivation for attention.

## Exercises

1. Explain in your own words what teacher forcing is, why it makes training easier, and what problem it causes at generation time.
2. With a vocabulary of 10,000 words, how many different output sentences of length 5 exist? How many candidates does beam search with $k = 4$ score at each step (before keeping the best 4)?
3. A decoder produces, for a 3-token output, the probabilities $0.5$, $0.4$ and $0.9$ for the chosen tokens. Compute the log-probability of the sentence (natural logarithm) and the length-normalized score (divide by 3).
4. In the code above, change `greedy_decode` so that it also returns the sum of the log-probabilities of the chosen tokens. (Hint: apply softmax to `W_out @ s` and take the log of the chosen entry.)

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| sequence-to-sequence model | модель «последовательность в последовательность» | 시퀀스-투-시퀀스 모델 |
| encoder | кодировщик | 인코더 |
| decoder | декодировщик | 디코더 |
| encoder–decoder architecture | архитектура «кодировщик–декодировщик» | 인코더–디코더 구조 |
| context vector | вектор контекста | 문맥 벡터 |
| special token | специальный токен | 특수 토큰 |
| end-of-sequence token | токен конца последовательности | 시퀀스 종료 토큰 |
| conditional language model | условная языковая модель | 조건부 언어 모델 |
| parallel corpus | параллельный корпус | 병렬 말뭉치 |
| teacher forcing | метод teacher forcing | 교사 강요 |
| exposure bias | смещение экспозиции | 노출 편향 |
| greedy decoding | жадное декодирование | 탐욕적 디코딩 |
| beam search | лучевой поиск | 빔 서치 |
| machine translation | машинный перевод | 기계 번역 |
| information bottleneck | информационное узкое место | 정보 병목 |

---
[← Previous: LSTM and GRU](12-lstm-and-gru.md) · [Contents](README.md) · [Next: The Idea of Attention (Bahdanau, Luong) →](14-the-idea-of-attention.md)
