# Chapter 20. Building a Tiny Transformer in PyTorch

> **Part 4 — The Transformer** · Level: Intermediate
>
> **In this chapter you will learn:** how to assemble embeddings, attention blocks and an output head into a working GPT-style model · how to train it with next-token prediction and read its loss curve · how to generate text from it, and what separates this toy from a real LLM

## Can 100,000 parameters learn to write?

In Chapters 16–19 we studied the Transformer piece by piece: multi-head attention with a causal mask, positional information, the block with residuals, normalization and an FFN, and the decoder-only architecture. Now we put everything together and ask a concrete question: can a model with about 100,000 parameters, trained for 15 seconds on a laptop CPU, learn to produce a Russian sentence character by character?

We will build a tiny decoder-only model in the spirit of GPT — let us call it **TinyGPT** — in about 65 lines of PyTorch. The program is shown below in four parts: data, model, training and generation. **Put the four code blocks one after another into a single file**, for example `tiny_gpt.py`, and run it with `python tiny_gpt.py`. It needs only PyTorch and runs on CPU. (On Windows, if printing Cyrillic characters raises an encoding error, set the environment variable `PYTHONIOENCODING=utf-8` first.)

The toy model is deliberately small, but it is not a simplification of the *idea*: the same pieces, scaled up by many orders of magnitude, with a better tokenizer and much more data, form a real LLM.

## Step 1: data and a character tokenizer

To keep everything visible, we use a **character-level** tokenizer: every distinct character is one token. As we saw in Chapter 3, real models use subword tokenizers such as BPE, but a character vocabulary needs no training and makes it easy to see what the model has learned. Our training text is two Russian sentences repeated 20 times: *«модель читает текст и предсказывает следующий символ»* ("the model reads text and predicts the next character") and *«внимание помогает модели смотреть на прошлые символы»* ("attention helps the model look at past characters").

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
text = ("модель читает текст и предсказывает следующий символ. "
        "внимание помогает модели смотреть на прошлые символы. ") * 20
chars = sorted(set(text))                          # character-level vocabulary
stoi = {c: i for i, c in enumerate(chars)}         # character -> id
itos = {i: c for c, i in stoi.items()}             # id -> character
data = torch.tensor([stoi[c] for c in text])
V, T, D, H, L = len(chars), 32, 64, 4, 2           # vocab, context, width, heads, layers
```

The vocabulary has $V = 26$ symbols: Cyrillic letters, the space and the period. The whole text becomes a 1-dimensional tensor `data` of token ids. The **hyperparameters** are deliberately tiny: a **context window** of $T = 32$ characters, model width $D = 64$, $H = 4$ heads (so each head has $d_k = 16$), and $L = 2$ blocks. For comparison, GPT-2 small has width 768, 12 heads and 12 blocks.

## Step 2: the model

The block follows Chapter 18 in its Pre-Norm form: LayerNorm → causal multi-head attention → residual, then LayerNorm → FFN → residual. For attention we now use PyTorch's built-in `F.scaled_dot_product_attention`, which computes exactly the formula of Chapter 16; the flag `is_causal=True` applies the causal mask for us. The FFN is the classic one, with hidden size $4D$ and GELU.

```python
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.ffn = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
    def forward(self, x):
        B, t, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q, k, v = [z.view(B, t, H, D // H).transpose(1, 2) for z in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # masked attention
        x = x + self.proj(y.transpose(1, 2).reshape(B, t, D))        # residual 1
        return x + self.ffn(self.ln2(x))                             # residual 2

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)              # token embeddings
        self.pos = nn.Embedding(T, D)              # learned positions (Chapter 17)
        self.blocks = nn.Sequential(*[Block() for _ in range(L)])
        self.ln_f = nn.LayerNorm(D)                # final normalization
        self.head = nn.Linear(D, V, bias=False)    # LM head: vector -> logits
        self.head.weight = self.tok.weight         # weight tying
        nn.init.normal_(self.tok.weight, std=0.02) # small init: logits start near 0
    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1)))
        return self.head(self.ln_f(self.blocks(x)))
```

The full model reads top to bottom like the diagram at the end of Chapter 18. Token ids become vectors through the embedding table (Chapter 5); learned position vectors are added; two blocks process the sequence; a final LayerNorm is applied; and the **language model head**, a linear layer of size $D \to V$, turns each vector into $V$ **logits**, unnormalized scores for the next token.

Two details deserve explanation.

**Weight tying.** The head matrix has shape $V \times D$, exactly like the embedding table. Many models, including GPT-2, use the *same* matrix for both. This **weight tying** saves parameters and has an intuitive meaning: the logit of token $j$ is the dot product of the final vector with the embedding of $j$, so the model predicts tokens whose embeddings point in the same direction as its output.

**Initialization.** PyTorch initializes `nn.Embedding` with standard deviation 1. With tying, this makes the initial logits large, and the first loss is around 45 instead of the expected $\ln V$. Why $\ln V$? A model that knows nothing should give every token probability $1/V$, and the cross-entropy of that guess is $-\ln(1/V) = \ln 26 \approx 3.26$. Re-initializing the embeddings with a small standard deviation of 0.02, as GPT-2 does, brings us to this sensible starting point. Checking that the first loss is close to $\ln V$ is a good habit for any classifier.

The model has **103,808 parameters**: each block has $12{,}480$ (QKV) $+ 4{,}160$ (output projection) $+ 33{,}088$ (FFN) $+ 256$ (two LayerNorms) $= 49{,}984$; two blocks give $99{,}968$; the position table adds $32 \cdot 64 = 2{,}048$, the token table $26 \cdot 64 = 1{,}664$ (shared with the head), and the final LayerNorm $128$.

## Step 3: training with next-token prediction

Training is the loop of Chapters 8–10. At each step we cut 16 random windows of $T + 1$ characters from the text. The first $T$ characters are the input `x`; the same window shifted by one position is the target `y`. Thanks to the causal mask, one window gives $T = 32$ training examples at once: position 0 predicts character 1, position 1 predicts character 2 from characters 0–1, and so on. The loss is the cross-entropy (Chapter 8) between the logits and the true next characters, averaged over all $16 \times 32$ positions, and the optimizer is AdamW (Chapter 9).

```python
model = TinyGPT()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
print(sum(p.numel() for p in model.parameters()), "parameters")

for step in range(1001):
    ix = torch.randint(len(data) - T - 1, (16,))             # 16 random windows
    x = torch.stack([data[i:i + T] for i in ix])
    y = torch.stack([data[i + 1:i + T + 1] for i in ix])     # targets: shifted by one
    loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 200 == 0:
        print(step, round(loss.item(), 3))
```

On our machine (PyTorch 2, CPU) the output was:

```
103808 parameters
0 3.27
200 2.955
400 1.439
600 0.398
800 0.109
1000 0.088
```

Exact numbers may differ slightly between PyTorch versions, but the shape of the curve is instructive. It starts at $3.27 \approx \ln 26$: a uniform guess. Then it stays near $2.95$ for a while. This value is not random: the entropy of the character frequencies of our text is $2.946$. At this stage the model has learned *which characters are common* (spaces, «о», «е») but not yet how to use the context — it is a unigram model in the sense of Chapter 4. Then attention "switches on", the loss falls quickly, and by step 1000 it is below $0.1$: the model predicts almost every next character correctly. Such plateaus followed by sudden drops are common when a model has to discover a new mechanism, here the ability to look back and copy.

## Step 4: generating text

Generation is autoregressive, as described in Chapter 1: give the model a prompt, take the logits at the **last** position, turn them into probabilities with softmax, **sample** one character, append it, and repeat. Because the position table has only $T$ rows, we feed at most the last $T$ characters (`idx[:, -T:]`). The decorator `@torch.no_grad()` turns off gradient tracking, since we are not training.

```python
@torch.no_grad()
def generate(prompt, n=60):
    idx = torch.tensor([[stoi[c] for c in prompt]])
    for _ in range(n):
        logits = model(idx[:, -T:])[:, -1, :]                # last position only
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return "".join(itos[i] for i in idx[0].tolist())

model.eval()
print(generate("модель "))
```

Our run printed `модель читает текст и предсказывает следующий символ. внимание помо` — a perfect continuation. For comparison, after only 300 steps (loss about 2) the same code printed something like `модель  пыотелюыемваей и пыом.уыи см.де`: Russian-looking letters with the right frequencies, but no words. The prompt may contain only characters from the vocabulary; any other character raises a `KeyError`.

## What separates TinyGPT from a real LLM

Be honest about what we have built: the model has **memorized** two sentences. With a text repeated 20 times there is nothing to generalize, and we did not even keep a validation set (Chapter 10) to measure it. Yet the architecture is the real one. To turn TinyGPT into a model like those in Part 5, one would change:

- **Tokenizer:** a BPE or SentencePiece vocabulary of tens of thousands of subwords instead of 26 characters (Chapter 3).
- **Data and scale:** billions of tokens of diverse text, and width, depth and number of heads larger by orders of magnitude (Chapters 26–28).
- **Modern components:** RoPE instead of learned positions (Chapter 17), RMSNorm and SwiGLU (Chapter 18), and often grouped-query attention (Chapter 25).
- **Training details:** dropout or other regularization, a learning-rate warm-up and decay, gradient clipping, mixed precision (Chapters 10 and 28).
- **Generation:** temperature, top-k and top-p sampling (Chapter 24) and a KV cache so that past keys and values are not recomputed (Chapter 36).

None of these changes the core loop you have just run: embed, attend, transform, predict the next token, compute cross-entropy, update. Part 5 builds on exactly this foundation.

## Summary

- A GPT-style model is: token embeddings + positions → a stack of causal Transformer blocks → final normalization → a linear head producing logits over the vocabulary.
- With a causal mask, one window of $T$ tokens provides $T$ next-token training examples, with targets equal to the input shifted by one.
- A well-initialized model starts with loss $\approx \ln V$; a plateau at the unigram entropy means the model knows character frequencies but not yet the context.
- Weight tying shares the embedding matrix with the output head and saves parameters.
- Generation repeatedly feeds the last $T$ tokens, samples from the softmax at the last position, and appends the result.

## Exercises

1. Explain in your own words why the training targets `y` are the inputs `x` shifted by one position, and why the causal mask is essential for this to work.
2. Compute the initial loss you would expect with a vocabulary of 50,000 tokens. Why is a first loss of about 45 a sign of a problem?
3. Remove the line `self.pos = ...` and the position term in `forward`, and retrain. What happens to the loss and to the generated text? Explain using Chapter 17.
4. Replace the training text with a paragraph of at least 2,000 characters of real Russian or Korean prose, keep 10% at the end as a validation set, and print the validation loss every 200 steps. When does the model start to overfit?

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| character-level tokenizer | посимвольный токенизатор | 문자 수준 토크나이저 |
| vocabulary | словарь | 어휘 |
| hyperparameter | гиперпараметр | 하이퍼파라미터 |
| context window | контекстное окно | 문맥 창 |
| language model head | голова языковой модели | 언어 모델 헤드 |
| logits | логиты | 로짓 |
| weight tying | связывание весов | 가중치 공유 |
| initialization | инициализация | 초기화 |
| cross-entropy | перекрёстная энтропия | 교차 엔트로피 |
| training loop | цикл обучения | 학습 루프 |
| plateau | плато | 정체 구간 |
| autoregressive generation | авторегрессионная генерация | 자기회귀 생성 |
| sampling | сэмплирование | 샘플링 |
| memorization | запоминание | 암기 |

---
[← Previous: Encoder, Decoder and Encoder–Decoder Architectures](19-transformer-architectures.md) · [Contents](README.md) · [Next: Pre-training Objectives: Causal and Masked Language Modeling →](21-pretraining-objectives.md)
