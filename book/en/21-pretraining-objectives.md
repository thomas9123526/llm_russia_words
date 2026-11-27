# Chapter 21. Pre-training Objectives: Causal and Masked Language Modeling

> **Part 5 — Language Models in Practice** · Level: Intermediate
>
> **In this chapter you will learn:** how a model learns from raw text without human labels · the two classic objectives, causal and masked language modeling · how other objectives (span corruption, replaced token detection) relate to them

## Two games with one sentence

In Chapter 20 we built a tiny Transformer, but we left an important question open: *what should it learn?* Suppose you have a huge collection of text — books, articles, web pages — and no labels at all. Nobody has marked which sentences are positive or negative, which words are names, or what the answers to questions are. Can a network still learn something useful from this?

Take the sentence *"the cat sat on the mat"* and play two games with it.

- **Game 1 — guess the next word.** I show you *"the cat sat on the"* and you guess *"mat"*. Then I show you *"the cat sat on"* and you guess *"the"*, and so on.
- **Game 2 — fill in the blank.** I show you *"the cat [?] on the mat"* and you guess *"sat"*.

In both games the correct answer is already in the text. We only hid part of it. This simple trick is called **self-supervised learning**: the training labels are produced automatically from the data itself. Because no human annotation is needed, we can train on billions of sentences. This stage is called **pre-training**, and the game we choose is the **pre-training objective**. After pre-training, the model is usually adapted to a specific **downstream task**, which we will discuss in Chapter 31.

Game 1 is **causal language modeling** (CLM), used by GPT and most modern LLMs. Game 2 is **masked language modeling** (MLM), introduced with BERT. This chapter explains both and the variants that grew out of them.

## Causal language modeling: predict the next token

Causal language modeling is exactly the language model of Chapter 1, now implemented with a Transformer. The model reads tokens from left to right and, at every position, predicts a probability distribution over the next token. The word *causal* means that a prediction at position $t$ may depend only on the past, never on the future — just as a cause comes before its effect.

Recall the chain rule of probability from Chapter 6. The probability of a sequence $x = (x_1, \dots, x_T)$ can be written as a product of conditional probabilities:

$$P(x_1, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})$$

Here $x_t$ is the token at position $t$ and $x_{<t}$ means all tokens before it. A model with parameters $\theta$ gives us $P_\theta(x_t \mid x_{<t})$, and training minimizes the negative log-likelihood:

$$\mathcal{L}_{\text{CLM}}(\theta) = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$

This is just the cross-entropy loss from Chapter 8, summed over positions. Its average per token, exponentiated, is the perplexity we met in Chapter 4.

How do we compute all $T$ predictions efficiently? We feed the whole sequence into the Transformer at once and use the **causal mask** from Chapter 16, so that position $t$ can only attend to positions $1, \dots, t$. The input is the sequence without its last token, and the target is the same sequence shifted one step to the left:

| position | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| input | the | cat | sat | on | the |
| target | cat | sat | on | the | mat |

One forward pass gives five predictions and five loss terms. This is an important practical advantage: **every token in the corpus provides a training signal**. Nothing is wasted.

The second advantage is that the training task is the same as the way the model is used. To generate text, we predict a token, append it to the input, and predict again (Chapter 24). A model trained with CLM has practised exactly this game. The disadvantage is that each representation sees only left context. When the model processes the word *"bank"* in *"the bank of the river"*, the hidden state at *"bank"* cannot yet look at *"river"*.

## Masked language modeling: fill in the blanks

Masked language modeling removes this limitation. We corrupt the input by hiding some tokens and ask the model to reconstruct them. Since nothing is hidden from the model's point of view except the masked tokens, the encoder can use **bidirectional context** — words on both the left and the right. There is no causal mask; every token attends to every other token (Chapter 19 called this an encoder architecture).

In BERT (Devlin et al., 2019), 15% of the token positions are selected for prediction. Then a small rule is applied to each selected position:

- 80% of the time the token is replaced by a special **mask token** `[MASK]`;
- 10% of the time it is replaced by a random token;
- 10% of the time it is left unchanged.

Why not always use `[MASK]`? Because `[MASK]` never appears when the model is later fine-tuned on real sentences. If the model only ever had to predict at `[MASK]` positions, it could learn to produce good representations only there. The random and unchanged cases force the model to keep a useful representation for *every* token, because any token might be one it must check.

Let $M$ be the set of selected positions and $\tilde{x}$ the corrupted sequence. The loss is

$$\mathcal{L}_{\text{MLM}}(\theta) = -\sum_{i \in M} \log P_\theta(x_i \mid \tilde{x})$$

Notice two differences from CLM. First, the condition is the whole corrupted sequence $\tilde{x}$, not only the past. Second, the sum runs only over $M$. With a 15% rate, a sentence of 100 tokens gives only about 15 loss terms, while CLM would give about 100. MLM therefore extracts less signal from each pass over the data.

There is also a theoretical subtlety. The masked tokens are predicted independently of each other. If *"New"* and *"York"* are both masked, the model predicts each one without knowing the other. So MLM does not define a proper probability for a whole sentence, and a model trained this way cannot easily generate text left to right. MLM models are excellent *readers* (classification, tagging, search) but poor *writers*. We will see this in detail in Chapter 22.

## Code: building training examples

The code below shows how training examples for both objectives are created from the same token ids. The labels value `-100` is the PyTorch convention for "ignore this position in the loss".

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
vocab_size, MASK_ID = 100, 3            # toy vocabulary; id 3 plays the role of [MASK]
tokens = torch.tensor([[12, 45, 7, 88, 23, 61, 9, 30]])   # one sentence, 8 token ids

# Causal LM: the input is tokens[:-1], the target is tokens[1:]
clm_input, clm_target = tokens[:, :-1], tokens[:, 1:]
print("CLM input :", clm_input.tolist())
print("CLM target:", clm_target.tolist())

# Masked LM: choose a fraction p of positions, then apply the 80/10/10 rule
def mask_tokens(x, p=0.15):
    labels = x.clone()
    chosen = torch.rand(x.shape) < p
    labels[~chosen] = -100                          # -100 = ignored by the loss
    x = x.clone()
    r = torch.rand(x.shape)
    x[chosen & (r < 0.8)] = MASK_ID                 # 80%: replace with [MASK]
    rand_pos = chosen & (r >= 0.8) & (r < 0.9)      # 10%: replace with a random token
    x[rand_pos] = torch.randint(4, vocab_size, (int(rand_pos.sum()),))
    return x, labels                                # remaining 10%: keep unchanged

mlm_input, mlm_labels = mask_tokens(tokens, p=0.3)  # large p so the toy example shows masks
print("MLM input :", mlm_input.tolist())
print("MLM labels:", mlm_labels.tolist())

logits = torch.randn(1, 8, vocab_size)              # stand-in for model outputs
loss = F.cross_entropy(logits.view(-1, vocab_size), mlm_labels.view(-1), ignore_index=-100)
print("MLM loss:", round(loss.item(), 3))           # averaged over labelled positions only
```

With this random seed, the masked input is `[12, 45, 3, 3, 23, 61, 9, 30]` and the labels are `[-100, -100, 7, 88, -100, -100, -100, -100]`: two positions were masked, and only they contribute to the loss. The loss on random logits is about 4.7, close to $\ln 100 \approx 4.6$, the value for uniform guessing over 100 tokens — a useful sanity check.

## Beyond the two classics

Many later objectives can be understood as variations of these two games.

- **Span corruption** (T5, Raffel et al., 2020). Instead of single tokens, whole spans of consecutive tokens are replaced by one sentinel token each, for example *"the cat `<X>` the mat"*. An encoder reads the corrupted text, and a decoder generates the missing spans: *"`<X>` sat on"*. This keeps bidirectional reading in the encoder but trains a real generator in the decoder.
- **Denoising** (BART, Lewis et al., 2020). The input is corrupted in several ways — masking, deleting, shuffling sentences — and an encoder–decoder model reconstructs the original text. It is a **denoising autoencoder** for language.
- **Prefix language modeling.** The first part of the sequence (the prefix) is read bidirectionally, and the rest is generated causally. One mask matrix allows full attention inside the prefix and causal attention after it.
- **Replaced token detection** (ELECTRA, Clark et al., 2020). A small generator fills in masked positions with plausible tokens, and the main model must decide, for *every* token, whether it is original or replaced. Because the loss covers all positions, it uses the data more efficiently than MLM.

| Objective | Context seen | Positions with loss | Can generate? |
|---|---|---|---|
| Causal LM | left only | all | yes |
| Masked LM | both sides | ~15% | not directly |
| Span corruption | both (encoder) | masked spans | yes (decoder) |
| Replaced token detection | both sides | all | no |

Why did causal LM become the standard for large models? There are three main reasons. It uses every token. Its training task matches generation, so one model can both understand and write. And, as we will see in Chapter 23, a sufficiently large causal model can solve new tasks just from instructions and examples in its input, without any change to its weights. The bidirectional advantage of MLM still matters for smaller models used for search and classification, which is why encoder models remain widely used.

## Summary

- Pre-training is self-supervised: we hide part of the text and train the model to recover it, so no human labels are needed.
- Causal language modeling predicts each token from the previous ones, $-\sum_t \log P_\theta(x_t \mid x_{<t})$, using a causal mask and shifted targets.
- Masked language modeling hides about 15% of tokens (80/10/10 rule) and predicts them from bidirectional context; the loss covers only the masked positions.
- CLM uses every token and matches generation; MLM gives richer bidirectional representations but cannot easily generate.
- Span corruption, denoising, prefix LM and replaced token detection are variations that combine the strengths of both.

## Exercises

1. Explain in your own words why BERT does not always replace a selected token with `[MASK]`.
2. A corpus has 10 billion tokens. How many loss terms does one pass give for CLM? For MLM with a 15% masking rate? What does this suggest about data efficiency?
3. Modify `mask_tokens` so that it masks whole words: if a word is split into several subword tokens (Chapter 3), all its pieces are masked together. What information do you need as an extra input?
4. Write the attention mask (a $5 \times 5$ matrix of 0s and 1s) for prefix language modeling where the first 2 tokens are the prefix.

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| pre-training | предобучение | 사전 학습 |
| pre-training objective | задача предобучения | 사전 학습 목표 |
| self-supervised learning | самообучение | 자기 지도 학습 |
| causal language modeling | каузальное языковое моделирование | 인과적 언어 모델링 |
| masked language modeling | маскированное языковое моделирование | 마스크 언어 모델링 |
| next-token prediction | предсказание следующего токена | 다음 토큰 예측 |
| causal mask | каузальная маска | 인과적 마스크 |
| mask token | токен маски | 마스크 토큰 |
| bidirectional context | двунаправленный контекст | 양방향 문맥 |
| cross-entropy | перекрёстная энтропия | 교차 엔트로피 |
| span corruption | маскирование фрагментов текста | 스팬 손상 |
| denoising autoencoder | шумоподавляющий автокодировщик | 잡음 제거 오토인코더 |
| downstream task | целевая задача | 다운스트림 과제 |

---
[← Previous: Building a Tiny Transformer in PyTorch](20-tiny-transformer.md) · [Contents](README.md) · [Next: BERT and Encoder Models →](22-bert-and-encoders.md)
