# Chapter 25. Modern Open Models: LLaMA, GQA, Mistral, Qwen

> **Part 5 — Language Models in Practice** · Level: Intermediate
>
> **In this chapter you will learn:** the "LLaMA recipe" that most open decoder models share · how grouped-query attention reduces the memory of the KV cache · what Mistral's sliding window attention and Qwen's large vocabulary add to the recipe

## Reading a configuration file

When you download an open model, one of the first files you see is its configuration. A typical one looks roughly like this (the numbers are illustrative, not those of one specific model):

```json
{
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 14336,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-05,
  "rope_theta": 10000.0,
  "vocab_size": 32000,
  "max_position_embeddings": 8192,
  "tie_word_embeddings": false
}
```

After Chapters 16–23 you can already read most of it: 32 layers of width 4096, 32 attention heads, a vocabulary of 32 thousand tokens. But some fields are new. Why are there only 8 *key–value* heads for 32 attention heads? Why is the activation `silu` and the normalization `rms_norm`? What is `rope_theta`? These fields describe a set of design choices that most open models share. In this chapter we go through that shared recipe and then look at what three influential families — LLaMA, Mistral and Qwen — contributed.

A note on the word "open". Most of these models are **open-weight models**: the trained parameters can be downloaded, but the training data and the full training code are usually not published, and the licences differ from model to model. A fully open-source model in the strict sense would also release data and code. Always read the licence before using a model.

## The LLaMA recipe

**LLaMA** (Touvron et al., 2023, Meta) was a family of decoder-only models trained only on publicly available data. Its paper described clearly a set of modifications to the GPT-2 architecture of Chapter 23. These choices were not new individually, but their combination became the de facto standard for open models:

1. **Pre-normalization with RMSNorm** (Chapter 18). Each sub-layer normalizes its input with
   $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot g$$
   where $d$ is the hidden size, $\epsilon$ is a small constant for numerical stability (the `rms_norm_eps` field) and $g$ is a learned scale vector. Compared with LayerNorm, there is no mean subtraction and no shift, which makes it slightly cheaper and works equally well in practice.
2. **SwiGLU feed-forward network** (Chapter 18):
   $$\text{FFN}(x) = W_2\big(\text{SiLU}(W_1 x) \odot W_3 x\big)$$
   Here $W_1$ and $W_3$ map from $d$ to the hidden size $d_{ff}$, $W_2$ maps back, and $\odot$ is element-wise multiplication. There are three matrices instead of two, so $d_{ff}$ is reduced to about $\tfrac{8}{3}d$ instead of $4d$, keeping the parameter count similar. This is why `intermediate_size` is not a simple multiple of 4.
3. **Rotary position embeddings, RoPE** (Chapter 17), applied to queries and keys in every layer instead of adding learned positions to the input. `rope_theta` is the base of the rotation frequencies; larger values are used to support longer contexts.
4. **No bias terms** in most linear layers, and a SentencePiece BPE tokenizer (Chapter 3).

Another influential decision concerned training length. The scaling-law analysis we will study in Chapter 27 suggests how to divide a fixed *training* budget between model size and data. LLaMA deliberately trained relatively small models on much more data than that analysis recommends, because a smaller model is cheaper every time it is *used*. For models that will serve many users, inference cost matters as much as training cost.

Later versions kept this core and changed details: longer contexts, more training data, larger vocabularies, and grouped-query attention, the subject of the next section.

## Grouped-query attention

Recall from Chapter 23 that during generation the model stores the keys and values of all previous tokens — the **KV cache** — so that it does not recompute them at every step (Chapter 36 covers this in detail). Its size is

$$\text{memory} = 2 \cdot L \cdot H_{kv} \cdot d_h \cdot T \cdot b$$

where the factor 2 counts keys and values, $L$ is the number of layers, $H_{kv}$ the number of key–value heads, $d_h$ the head dimension, $T$ the number of cached tokens and $b$ the bytes per number (2 for 16-bit floats). With $L = 32$, $H_{kv} = 32$, $d_h = 128$ and $T = 4096$, this is exactly 2 GiB *for one sequence*. Serving many users with long contexts quickly fills a GPU, and reading this cache at every step makes generation limited by memory bandwidth rather than by computation.

In standard **multi-head attention** (MHA, Chapter 16) each of the $H$ query heads has its own key and value head, so $H_{kv} = H$. **Multi-query attention** (MQA, Shazeer, 2019) goes to the other extreme: all query heads share a single key and value head, $H_{kv} = 1$. This shrinks the cache by a factor of $H$ but can reduce quality and make training less stable.

**Grouped-query attention** (GQA, Ainslie et al., 2023) is the compromise. The $H$ query heads are divided into $H_{kv}$ groups, and each group shares one key head and one value head. With 32 query heads and 8 KV heads, every 4 query heads share the same keys and values. The cache shrinks by a factor of $H / H_{kv} = 4$, and experiments showed that quality stays close to MHA. Ainslie et al. also showed that an existing MHA checkpoint can be converted to GQA by averaging the key and value heads within each group and training briefly — they called this *uptraining*.

```python
import torch
import torch.nn.functional as F

def gqa(x, Wq, Wk, Wv, n_heads, n_kv_heads):
    B, T, d = x.shape
    hd = d // n_heads                                         # head dimension
    q = (x @ Wq).view(B, T, n_heads, hd).transpose(1, 2)      # (B, H, T, hd)
    k = (x @ Wk).view(B, T, n_kv_heads, hd).transpose(1, 2)   # (B, H_kv, T, hd)
    v = (x @ Wv).view(B, T, n_kv_heads, hd).transpose(1, 2)
    group = n_heads // n_kv_heads                             # query heads per K/V head
    k = k.repeat_interleave(group, dim=1)                     # each K/V head is shared
    v = v.repeat_interleave(group, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return out.transpose(1, 2).reshape(B, T, d)

torch.manual_seed(0)
d, H, H_kv = 64, 8, 2
x = torch.randn(1, 5, d)                                      # 5 tokens
Wq = torch.randn(d, d) / d**0.5
Wk = torch.randn(d, H_kv * d // H) / d**0.5                   # K, V projections: 4x smaller
Wv = torch.randn(d, H_kv * d // H) / d**0.5
print(gqa(x, Wq, Wk, Wv, H, H_kv).shape)                      # torch.Size([1, 5, 64])

def kv_cache_gib(layers, kv_heads, head_dim, tokens, bytes_per=2):  # 2 bytes = fp16
    return 2 * layers * kv_heads * head_dim * tokens * bytes_per / 2**30  # 2 = K and V

for kv in (32, 8, 1):                                         # MHA, GQA, MQA
    print(f"{kv:2d} KV heads: {kv_cache_gib(32, kv, 128, 4096):.4f} GiB")
```

The output shows 2 GiB for MHA, 0.5 GiB for GQA with 8 KV heads and 0.0625 GiB for MQA. Notice that `repeat_interleave` only exists at computation time: what is *stored* in the cache are the small `k` and `v` tensors before repetition. The key and value projection matrices are also smaller, so GQA saves a few parameters as well.

## Mistral: sliding window attention

**Mistral 7B** (Jiang et al., 2023) used the LLaMA-style recipe with GQA and added **sliding window attention** (SWA). In ordinary causal attention token $i$ attends to all tokens $j \le i$. With a window of size $W$, it attends only to the last $W$ tokens: $i - W < j \le i$. For $T = 6$ and $W = 3$ the mask is (1 = allowed):

```text
     j: 1 2 3 4 5 6
i = 1:  1 0 0 0 0 0
i = 2:  1 1 0 0 0 0
i = 3:  1 1 1 0 0 0
i = 4:  0 1 1 1 0 0
i = 5:  0 0 1 1 1 0
i = 6:  0 0 0 1 1 1
```

Does the model then forget everything older than $W$ tokens? Not completely. In layer 1, token $i$ sees up to $W$ tokens back. But those tokens, in their own layer-1 computation, already saw $W$ tokens further back. After $L$ layers, information can travel up to about $L \times W$ positions — just as stacked convolutions in image networks see a larger region in deeper layers. This number is called the *receptive field*.

The practical gain is in the cache. Since no token ever needs keys and values older than $W$ positions, the cache can be a fixed-size **rolling buffer cache**: position $i$ is stored in slot $i \bmod W$, overwriting the oldest entry. Memory no longer grows with sequence length. The Mistral team later also released **Mixtral**, a mixture-of-experts model, which we will study in Chapter 29. The models were published under a permissive licence, which helped them spread widely.

## Qwen and the role of the vocabulary

**Qwen** (Bai et al., 2023, Alibaba) is another large family of open models, released in many sizes, with later versions including code, mathematics, multimodal and mixture-of-experts variants. The architecture again follows the LLaMA recipe — RMSNorm, SwiGLU, RoPE — with small differences, for example bias terms in the query, key and value projections. Later generations also adopted GQA.

The most visible difference is the **vocabulary size**. Qwen uses a byte-level BPE vocabulary of roughly 150 thousand tokens, several times larger than the 32 thousand of the original LLaMA. A large vocabulary is a trade-off:

- **Advantage:** text in many languages, especially Chinese, but also Russian and Korean, is split into fewer tokens. Recall from Chapter 3 that a tokenizer's **fertility** — the average number of tokens per word — matters: fewer tokens mean that more text fits into the context window, generation needs fewer steps, and cost per sentence goes down.
- **Cost:** the embedding matrix and the output layer grow as $V \cdot d$. With $V = 150{,}000$ and $d = 4096$ this is about 0.6 billion parameters per matrix, and the softmax over the vocabulary becomes more expensive. Rare tokens also receive fewer training updates.

For small models, the embedding matrices can be a large share of all parameters, which is why small models often tie input and output embeddings (Chapter 23) while large ones usually do not. We will return to vocabulary and language balance in Chapter 30 on multilingual models.

## The shared picture

| Component | GPT-2 (Chapter 23) | Typical modern open model |
|---|---|---|
| Normalization | LayerNorm, pre-norm | RMSNorm, pre-norm |
| Feed-forward | GELU, $d_{ff} = 4d$ | SwiGLU, $d_{ff} \approx \tfrac{8}{3}d$ or larger |
| Positions | learned absolute | RoPE |
| Attention | MHA | GQA, sometimes sliding window |
| Biases | yes | mostly no |
| Vocabulary | ~50K BPE | 32K – 150K+ BPE |

The main lesson is that the Transformer of 2017 has changed surprisingly little. The progress of open models came mostly from better data, more training, longer context and efficiency improvements like GQA — topics of Parts 6 and 8.

## Summary

- Most open decoder models share the "LLaMA recipe": pre-norm RMSNorm, SwiGLU FFN, RoPE, no biases, BPE tokenizer.
- Open-weight models publish their parameters, but usually not their data; licences differ.
- The KV cache size is $2 L H_{kv} d_h T b$; grouped-query attention shares each key–value head among a group of query heads and reduces the cache by $H / H_{kv}$ with little loss of quality.
- Mistral's sliding window attention limits each layer to the last $W$ tokens; stacked layers still give a receptive field of about $L \times W$, and a rolling buffer keeps cache memory constant.
- Qwen's large vocabulary lowers the number of tokens per text for many languages at the cost of larger embedding and output matrices.

## Exercises

1. Explain in your own words why GQA reduces memory during generation but has little effect on the amount of computation during training.
2. Compute the KV cache size for a model with $L = 80$, $H = 64$, $H_{kv} = 8$, $d_h = 128$, context $T = 8192$, 16-bit numbers, for a batch of 4 sequences. How much would MHA need?
3. Modify `gqa` to accept a window size `W` and build an explicit boolean mask for sliding window attention instead of `is_causal=True`. Check that with `W` larger than `T` the output equals the causal version.
4. A tokenizer with $V = 32{,}000$ splits a Russian text into 1,500 tokens; one with $V = 150{,}000$ splits it into 1,000 tokens. With $d = 4096$, how many extra embedding parameters does the larger vocabulary cost, and what fraction of context length does it save?

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| open-weight model | модель с открытыми весами | 오픈 웨이트 모델 |
| RMSNorm | RMSNorm | RMSNorm |
| SwiGLU | SwiGLU | SwiGLU |
| rotary position embedding (RoPE) | ротационное позиционное кодирование (RoPE) | 회전 위치 임베딩 (RoPE) |
| KV cache | KV-кэш | KV 캐시 |
| multi-head attention | многоголовое внимание | 멀티헤드 어텐션 |
| multi-query attention (MQA) | внимание с общими ключами и значениями | 멀티쿼리 어텐션 (MQA) |
| grouped-query attention (GQA) | внимание с группировкой запросов | 그룹 쿼리 어텐션 (GQA) |
| sliding window attention | внимание со скользящим окном | 슬라이딩 윈도 어텐션 |
| rolling buffer cache | кольцевой буфер кэша | 롤링 버퍼 캐시 |
| receptive field | рецептивное поле | 수용 영역 |
| vocabulary size | размер словаря | 어휘 크기 |
| tokenizer fertility | фертильность токенизатора | 토크나이저 생산성 |
| memory bandwidth | пропускная способность памяти | 메모리 대역폭 |

---
[← Previous: Generating Text: Greedy, Beam, Temperature, Top-k, Top-p](24-text-generation.md) · [Contents](README.md) · [Next: Training Data: Collection, Cleaning and Deduplication →](26-training-data.md)
