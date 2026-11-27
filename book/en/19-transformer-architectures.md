# Chapter 19. Encoder, Decoder and Encoder–Decoder Architectures

> **Part 4 — The Transformer** · Level: Intermediate
>
> **In this chapter you will learn:** how the original Transformer combines an encoder and a decoder through cross-attention · how encoder-only and decoder-only models are built from the same blocks · which family suits which task, and why decoder-only models dominate today

## Three tasks, three shapes

Consider three tasks. (1) Decide whether a product review is positive or negative. (2) Translate a Russian sentence, *«Кошка спит на диване»*, into Korean, *"고양이가 소파에서 자고 있다"*. (3) Continue a text that begins "Once upon a time".

In the first task the whole input is available at once, and the output is a single label; the model may read every word in both directions. In the second task there is an input sequence and a *different* output sequence, produced word by word. In the third there is only one sequence, which the model extends to the right. These three shapes correspond to the three families of Transformer models: **encoder-only**, **encoder–decoder**, and **decoder-only**.

All three are built from the block of Chapter 18. They differ in only two things: *which mask* the self-attention uses, and *whether there is cross-attention* to a second sequence. Once you see this, the zoo of model names — BERT, GPT, T5 and others — becomes much simpler.

## The original Transformer: encoder and decoder

The Transformer of "Attention Is All You Need" was designed for machine translation, continuing the sequence-to-sequence line of Chapter 13. It has two stacks.

The **encoder** reads the source sentence. It is a stack of $N$ blocks with *unmasked* self-attention: every source token can attend to every other source token, to the left and to the right. Its output is a sequence of vectors $H = (h_1, \dots, h_n)$, one per source token, each enriched with the context of the whole sentence. This output is often called the **memory**.

The **decoder** produces the target sentence one token at a time. Each decoder block has *three* sublayers instead of two:

1. **masked self-attention** over the target tokens generated so far, with the causal mask of Chapter 16;
2. **cross-attention**, in which the queries come from the decoder and the keys and values come from the encoder memory;
3. the feed-forward network.

Each sublayer has its residual connection and normalization, as in Chapter 18. Cross-attention is the Transformer version of the Bahdanau attention of Chapter 14. If $Y$ is the matrix of decoder vectors (one row per target token), it computes

$$
\text{CrossAttention}(Y, H) = \text{softmax}\!\left(\frac{(Y W^Q)(H W^K)^\top}{\sqrt{d_k}}\right) H W^V .
$$

The only difference from self-attention is that queries and keys come from *different* sequences. The score matrix therefore has size $m \times n$, where $m$ is the target length and $n$ the source length. When the decoder is about to produce *소파에서* ("on the sofa"), its query can find the encoder vectors of *на диване*, even though the word order of Russian and Korean differs. No mask is needed in cross-attention, because the whole source is known in advance.

Training uses **teacher forcing**, as in Chapter 13: the decoder receives the correct target prefix (shifted by one position) and learns to predict each next token, all positions in parallel thanks to the causal mask. At inference time, the encoder runs once, and the decoder runs repeatedly, appending one generated token at a time.

## Encoder-only models

If we keep only the encoder stack, we get an **encoder-only model**. Its self-attention is bidirectional, so each output vector depends on the **bidirectional context** — both the words before and the words after. This is ideal for *understanding* tasks: classification, named-entity recognition, question answering by selecting a span, and producing sentence embeddings for search (Chapter 37).

The best-known example is BERT (Chapter 22). Since an encoder sees the whole input, it cannot be trained by simply predicting the next token — the answer would be visible. Instead, BERT uses **masked language modeling**: some input tokens are replaced by a special `[MASK]` token, and the model learns to recover them from both sides (Chapter 21). The price is that an encoder-only model is not a natural text generator: it has no built-in way to produce a long sequence from left to right.

## Decoder-only models

If we keep only the decoder stack and remove the cross-attention sublayer, we get a **decoder-only model**: blocks with causal self-attention and an FFN, nothing else. It is trained with the simplest possible objective, **next-token prediction** (causal language modeling): at every position $t$, predict token $t+1$ from tokens $1, \dots, t$. The GPT family (Chapter 23) and most modern LLMs, including LLaMA and its relatives (Chapter 25), have this architecture.

How can a decoder-only model translate, if it has no encoder? By treating the source as the beginning of the text. The input "Russian: Кошка спит на диване. Korean:" is a prefix, and the translation is its natural continuation. The causal self-attention over the prefix plays the role that cross-attention played in the encoder–decoder: the Korean tokens attend back to the Russian ones.

Why has this simple design won? Several reasons work together:

- **One uniform objective.** Every token of every text is a training example, so no labeled data or special masking scheme is needed.
- **Every task is text.** Classification, translation, summarization and dialogue can all be written as "continue this text", which leads to in-context learning (Chapter 23).
- **Simplicity at scale.** One stack of identical blocks is easy to scale, parallelize and optimize; scaling laws (Chapter 27) were first studied mostly on such models.
- **Efficient generation.** Keys and values of previous tokens never change under a causal mask, so they can be cached (the KV cache, Chapter 36).

## Encoder–decoder models today and the prefix LM

The encoder–decoder design did not disappear. T5 reformulated every NLP task as "text in, text out" with an encoder–decoder, and BART combined a bidirectional encoder with an autoregressive decoder. Such models remain strong for tasks with a clear input and output, such as translation and **summarization**, because the encoder can read the input bidirectionally and the decoder can focus on generation.

Between the two extremes there is a compromise: the **prefix language model** (prefix LM). It uses a single stack, like a decoder, but with a mixed mask: tokens of the prefix (the input) attend to each other bidirectionally, while tokens of the continuation attend causally. The code below builds a small encoder–decoder with PyTorch's built-in modules and prints the three masks side by side.

```python
import torch
import torch.nn as nn

T_src, T_tgt, d = 6, 4, 32
src = torch.randn(1, T_src, d)   # embedded source sentence (e.g. Russian)
tgt = torch.randn(1, T_tgt, d)   # embedded target prefix generated so far (e.g. Korean)

# Encoder: every token sees every token, so no mask is needed
enc_layer = nn.TransformerEncoderLayer(d, nhead=4, batch_first=True)
memory = nn.TransformerEncoder(enc_layer, num_layers=2)(src)

# Decoder: causal self-attention + cross-attention to the encoder output
dec_layer = nn.TransformerDecoderLayer(d, nhead=4, batch_first=True)
decoder = nn.TransformerDecoder(dec_layer, num_layers=2)
causal = nn.Transformer.generate_square_subsequent_mask(T_tgt)  # -inf above the diagonal
out = decoder(tgt, memory, tgt_mask=causal)
print(memory.shape, out.shape)

# Masks of the three families for 4 tokens (1 = may attend)
full = torch.ones(4, 4, dtype=torch.int)     # encoder-only (BERT)
causal01 = torch.tril(full)                  # decoder-only (GPT)
prefix = causal01.clone()
prefix[:, :2] = 1                            # prefix LM: the first 2 tokens are visible to all
print(full, causal01, prefix, sep="\n")
```

The first line of output is `torch.Size([1, 6, 32]) torch.Size([1, 4, 32])`: the memory has one vector per source token, the decoder output one vector per target token. Then come the three masks. The encoder mask is all ones. The decoder mask is lower-triangular. The prefix-LM mask equals the causal mask except for the first two columns: the first two tokens form the prefix and can see each other, and every later token can see the whole prefix.

To compare the families at a glance:

| | Encoder-only | Encoder–decoder | Decoder-only |
|---|---|---|---|
| Self-attention mask | full (bidirectional) | full in encoder, causal in decoder | causal |
| Cross-attention | no | yes | no |
| Typical objective | masked LM | sequence-to-sequence, span corruption | next-token prediction |
| Typical tasks | classification, tagging, embeddings | translation, summarization | generation, chat, almost everything |
| Examples | BERT | original Transformer, T5, BART | GPT, LLaMA |

## Summary

- All Transformer families use the same block; they differ in the self-attention mask and in the presence of cross-attention.
- The original Transformer is an encoder–decoder: a bidirectional encoder produces a memory, and a causal decoder reads it through cross-attention.
- In cross-attention the queries come from the decoder and the keys and values from the encoder output.
- Encoder-only models (BERT) see bidirectional context and suit understanding tasks; they are trained with masked language modeling.
- Decoder-only models (GPT, LLaMA) use causal attention and next-token prediction; their simplicity and generality made them the dominant LLM architecture.

## Exercises

1. Explain in your own words why an encoder-only model cannot be trained by next-token prediction.
2. A source sentence has 20 tokens and the target has 15. What are the shapes of the score matrices in the decoder's self-attention and in its cross-attention (per head)?
3. Write the mask of a prefix LM for 6 tokens with a prefix of length 3, and check it with code.
4. Describe how you would turn sentiment classification into a task for a decoder-only model. What would the input text and the expected continuation look like?

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| encoder | кодировщик | 인코더 |
| decoder | декодировщик | 디코더 |
| encoder–decoder architecture | архитектура «кодировщик-декодировщик» | 인코더–디코더 아키텍처 |
| encoder-only model | модель только с кодировщиком | 인코더 전용 모델 |
| decoder-only model | модель только с декодировщиком | 디코더 전용 모델 |
| cross-attention | перекрёстное внимание | 교차 어텐션 |
| memory (encoder output) | память (выход кодировщика) | 메모리(인코더 출력) |
| bidirectional context | двунаправленный контекст | 양방향 문맥 |
| teacher forcing | метод teacher forcing | 교사 강요 |
| masked language modeling | маскированное языковое моделирование | 마스크 언어 모델링 |
| next-token prediction | предсказание следующего токена | 다음 토큰 예측 |
| prefix language model | префиксная языковая модель | 접두사 언어 모델 |
| machine translation | машинный перевод | 기계 번역 |
| summarization | суммаризация | 요약 |

---
[← Previous: The Transformer Block: Residuals, LayerNorm/RMSNorm, FFN, GLU](18-transformer-block.md) · [Contents](README.md) · [Next: Building a Tiny Transformer in PyTorch →](20-tiny-transformer.md)
