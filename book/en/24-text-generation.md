# Chapter 24. Generating Text: Greedy, Beam, Temperature, Top-k, Top-p

> **Part 5 — Language Models in Practice** · Level: Intermediate
>
> **In this chapter you will learn:** why turning a probability distribution into text is a separate design decision · deterministic decoding with greedy and beam search · stochastic decoding with temperature, top-k and top-p sampling

## One distribution, many possible texts

Give a trained language model the prompt *"In the morning I drink a cup of"*. After the last Transformer block and the output layer (Chapter 23), the model returns one **logit** for each token of the vocabulary. Suppose that, restricted to five candidates, the logits and their softmax probabilities are:

| token | tea | coffee | water | milk | soup |
|---|---|---|---|---|---|
| logit $z_i$ | 2.0 | 1.0 | 0.5 | 0.1 | −1.0 |
| probability $p_i$ | 0.559 | 0.205 | 0.125 | 0.084 | 0.028 |

The model has done its job: it has given us a **probability distribution** over the next token. But it has not told us *which* token to write. Should we always take *"tea"*? Should we sometimes take *"coffee"*? Should we look further ahead before deciding? The rule we use is called the **decoding strategy**, and it is chosen at inference time, not during training. The same model can produce dull, repetitive text or creative, varied text — or nonsense — depending only on this rule.

There are two broad families: **deterministic** methods, which search for a likely sequence, and **stochastic** methods, which sample from the distribution. We will study both, and then see how they are combined in practice.

## Greedy decoding and beam search

**Greedy decoding** is the simplest rule: at every step take the most probable token,

$$x_t = \arg\max_{v} P_\theta(v \mid x_{<t})$$

In our example, greedy decoding writes *"tea"*. It is fast and deterministic — the same prompt always gives the same output — which is useful for tasks with one correct answer, such as short factual questions or code completion.

Greedy decoding has two problems. First, it is *short-sighted*. The best token now does not always lead to the best sequence. Consider a toy model with two steps. At step 1, $P(A) = 0.5$ and $P(B) = 0.4$. If we choose $A$, the best next token has probability $0.4$, so the sequence has probability $0.5 \times 0.4 = 0.20$. If we choose $B$, the best next token has probability $0.9$, giving $0.4 \times 0.9 = 0.36$. Greedy decoding picks $A$ and misses the better sequence.

Second, in open-ended text greedy decoding often falls into **repetition loops**: once a phrase has been produced, the model sees it in its context and finds it even more likely the next time, so the same sentence can repeat again and again.

**Beam search** addresses the first problem. Instead of one partial sequence, it keeps the $B$ best ones (the *beam*; $B$ is the **beam width**). At each step every sequence in the beam is extended by every possible token, the score of each candidate is computed, and only the $B$ best candidates survive. The score is the sum of log-probabilities:

$$\text{score}(x_{1:t}) = \sum_{s=1}^{t} \log P_\theta(x_s \mid x_{<s})$$

With $B = 2$ in our toy example, both $A$ and $B$ survive step 1, and at step 2 the search finds $B$ followed by its best token, with $\log 0.36 \approx -1.02$, better than $\log 0.20 \approx -1.61$. Greedy decoding is simply beam search with $B = 1$.

Because every log-probability is negative, longer sequences always have lower scores, so beam search prefers short outputs. A common fix is a **length penalty**: divide the score by $t^\alpha$, where $t$ is the length and $\alpha$ is typically between 0.5 and 1.

Beam search works well when the output is strongly determined by the input — machine translation, summarization, speech recognition. For open-ended generation, such as stories or dialogue, it tends to produce generic and repetitive text. Holtzman et al. (2020) showed that human text is *not* the most probable text: people regularly choose words that a model rates as only moderately likely. Maximizing probability is therefore the wrong goal for creative text. This leads us to sampling.

## Sampling and temperature

In **sampling** we draw the next token at random, in proportion to its probability. In our example, *"tea"* is chosen about 56% of the time, *"coffee"* 21%, and so on. Sampling produces varied text and naturally avoids exact repetition, but pure sampling has a weakness: the vocabulary has tens of thousands of tokens, and the many unlikely ones together can carry a noticeable share of probability. Sooner or later an absurd token is sampled, and the text goes off track.

The first control is **temperature** $T > 0$. We divide the logits by $T$ before the softmax:

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

Here $z_i$ is the logit of token $i$ and the sum runs over the vocabulary. With $T = 1$ we get the model's own distribution. With $T < 1$ the differences between logits are magnified and the distribution becomes sharper; as $T \to 0$ all the probability goes to the top token, and sampling becomes greedy decoding. With $T > 1$ the distribution becomes flatter; as $T \to \infty$ it approaches uniform. For our logits:

| $T$ | tea | coffee | water | milk | soup |
|---|---|---|---|---|---|
| 0.5 | 0.826 | 0.112 | 0.041 | 0.018 | 0.002 |
| 1.0 | 0.559 | 0.205 | 0.125 | 0.084 | 0.028 |
| 2.0 | 0.372 | 0.226 | 0.176 | 0.144 | 0.083 |

The name comes from statistical physics, where the same formula (the Boltzmann distribution) describes how a hotter system visits more states. Low temperature means conservative and focused text; high temperature means diverse but riskier text.

## Truncation: top-k and top-p

Temperature changes the *shape* of the distribution but never removes the long tail of bad tokens. **Truncation** methods cut the tail off before sampling.

**Top-k sampling** (used by Fan et al., 2018, for story generation) keeps only the $k$ most probable tokens, sets the others to zero, renormalizes, and samples. With $k = 2$ in our example, only *"tea"* and *"coffee"* remain, with probabilities $0.559 / 0.764 \approx 0.731$ and $0.269$. The weakness of top-k is that a fixed $k$ fits some contexts badly. After *"The capital of France is"* almost all probability is on one token and $k = 50$ admits 49 bad candidates. After *"My favourite food is"* there are hundreds of reasonable continuations and $k = 50$ cuts off many good ones.

**Top-p sampling**, also called **nucleus sampling** (Holtzman et al., 2020), solves this by fixing the probability mass instead of the number of tokens. Sort the tokens by probability and keep the smallest set $V_p$ whose total probability is at least $p$:

$$V_p = \text{smallest set such that } \sum_{v \in V_p} P(v \mid x_{<t}) \ge p$$

With $p = 0.8$: *"tea"* gives 0.559, adding *"coffee"* gives 0.764 (still less than 0.8), adding *"water"* gives 0.889 ≥ 0.8, so the nucleus is {tea, coffee, water}. After renormalization the probabilities are 0.629, 0.231 and 0.140. When the model is confident, the nucleus is small; when it is uncertain, the nucleus grows automatically.

In practice these controls are combined: temperature first, then top-k and/or top-p. Additional tools include a **repetition penalty**, which lowers the logits of tokens that have already appeared, and **stop sequences**, which end generation when a given string is produced. Typical settings for a chat assistant are a temperature below 1 together with top-p around 0.9; for code or factual answers, greedy decoding or a very low temperature is often preferred.

## Code: one function for all strategies

```python
import torch

def sample_next(logits, temperature=1.0, top_k=0, top_p=1.0):
    """Choose the next token id from a 1-D tensor of logits."""
    if temperature == 0:                                  # greedy decoding
        return int(torch.argmax(logits))
    logits = logits / temperature                         # sharpen (T<1) or flatten (T>1)
    if top_k > 0:                                         # keep only the k best tokens
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:                                       # nucleus: smallest set with mass >= p
        sorted_logits, idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        remove = torch.cumsum(probs, dim=-1) - probs >= top_p   # mass before token >= p
        sorted_logits[remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(0, idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1))

torch.manual_seed(0)
words = ["tea", "coffee", "water", "milk", "soup"]
logits = torch.tensor([2.0, 1.0, 0.5, 0.1, -1.0])
for T in (0.5, 1.0, 2.0):
    print("T =", T, [round(p, 3) for p in torch.softmax(logits / T, -1).tolist()])
print("greedy :", words[sample_next(logits, temperature=0)])
print("top-k 2:", [words[sample_next(logits, top_k=2)] for _ in range(8)])
print("top-p .8:", [words[sample_next(logits, top_p=0.8)] for _ in range(8)])
```

The line with `cumsum(probs) - probs` deserves attention. For each sorted token it gives the probability mass of all tokens *before* it. A token is removed only if the tokens before it already reach $p$; this guarantees that the token which crosses the threshold (here *"water"*) is kept, and that at least one token always survives. Running the code, greedy always prints *"tea"*, top-k samples only *"tea"* and *"coffee"*, and top-p also produces *"water"* but never *"milk"* or *"soup"*. In a full generator, `sample_next` is called once per step inside the loop from Chapter 23.

## Summary

- A language model outputs a distribution; the decoding strategy decides which token to write, and it is chosen at inference time.
- Greedy decoding takes the argmax at each step; it is fast and deterministic but short-sighted and prone to repetition.
- Beam search keeps the $B$ best partial sequences by total log-probability; with a length penalty it is a strong choice for translation and summarization, but produces bland open-ended text.
- Temperature rescales logits: $T < 1$ sharpens, $T > 1$ flattens, $T \to 0$ becomes greedy.
- Top-k keeps a fixed number of tokens, top-p keeps the smallest set with probability mass $p$; both remove the unreliable tail before sampling.

## Exercises

1. Explain in your own words why the most probable sequence is not necessarily the best text for a story, but often is the best for translation.
2. For the logits $(2.0, 1.0, 0.5, 0.1, -1.0)$, compute the nucleus for $p = 0.5$ and for $p = 0.95$. How many tokens does each contain?
3. Run the beam search toy example by hand with $B = 1$ and $B = 2$ and write down the scores of all candidates at step 2.
4. Extend `sample_next` with a repetition penalty: given a list of previously generated ids, divide their positive logits by a factor $r > 1$ (and multiply negative ones by $r$). Check that with $r = 2$ and history `[0]` the probability of *"tea"* decreases.

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| decoding strategy | стратегия декодирования | 디코딩 전략 |
| greedy decoding | жадное декодирование | 탐욕적 디코딩 |
| beam search | лучевой поиск | 빔 서치 |
| beam width | ширина луча | 빔 너비 |
| length penalty | штраф за длину | 길이 페널티 |
| sampling | сэмплирование | 샘플링 |
| temperature | температура | 온도 |
| top-k sampling | top-k сэмплирование | Top-k 샘플링 |
| nucleus sampling (top-p) | ядерное сэмплирование (top-p) | 핵 샘플링 (Top-p) |
| truncation | усечение | 절단 |
| repetition penalty | штраф за повторы | 반복 페널티 |
| probability distribution | распределение вероятностей | 확률 분포 |
| stop sequence | стоп-последовательность | 정지 시퀀스 |

---
[← Previous: GPT and Decoder-Only Models, In-Context Learning](23-gpt-and-decoders.md) · [Contents](README.md) · [Next: Modern Open Models: LLaMA, GQA, Mistral, Qwen →](25-modern-open-models.md)
