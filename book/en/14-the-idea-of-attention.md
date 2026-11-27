# Chapter 14. The Idea of Attention (Bahdanau, Luong)

> **Part 3 — Sequences and Attention** · Level: Intermediate
>
> **In this chapter you will learn:** how attention lets the decoder look back at every source word instead of one compressed vector · how scores, softmax weights and a weighted sum produce a context vector at each step · the additive (Bahdanau) and multiplicative (Luong) score functions and what attention weights tell us

## Looking back at the source

At the end of Chapter 13 we met the bottleneck: the encoder squeezes the whole source sentence into one vector $c$, and the decoder must produce every target word from that single summary. We also noticed how a human translator works differently. When translating "The black cat sat on the mat" into Russian, while writing «кошка» they look at "cat"; while writing «сидела» they look at "sat". At each moment their eyes are on a different part of the source.

This is the whole idea of **attention**. Instead of throwing away all encoder states except the last one, we keep them all: $h_1, h_2, \dots, h_S$, one vector per source token. Then, at every decoder step, we let the decoder decide which of these vectors are relevant right now and build a fresh context vector from them. The context vector is no longer the same for the whole sentence; there is a separate $c_t$ for every output step $t$.

The mechanism was introduced for machine translation by Bahdanau, Cho and Bengio (2015) in a paper with a telling title: "Neural Machine Translation by Jointly Learning to Align and Translate". It removed the length problem of Chapter 13 almost completely, and it became the seed from which the Transformer grew.

## Attention in three steps

Suppose the decoder is about to produce the target token at step $t$, and its current state is $s$ (we will discuss shortly whether this is $s_{t-1}$ or $s_t$). The encoder states are $h_1, \dots, h_S$. Attention computes the context vector in three steps.

**Step 1: scores.** For every source position $j$, compute a number $e_{j}$ that says how well the encoder state $h_j$ matches what the decoder needs now:

$$e_{j} = \operatorname{score}(s, h_j)$$

**Step 2: weights.** Turn the scores into a probability distribution with softmax (Chapter 7):

$$\alpha_{j} = \frac{\exp(e_{j})}{\sum_{k=1}^{S} \exp(e_{k})}$$

The **attention weights** $\alpha_j$ are positive and sum to 1. A large $\alpha_j$ means "look at source token $j$".

**Step 3: context.** Take the weighted sum of the encoder states:

$$c_t = \sum_{j=1}^{S} \alpha_{j}\, h_j$$

The context vector $c_t$ is a blend of the source states, dominated by the ones with the largest weights. The decoder then uses $c_t$, together with its own state, to predict the next token.

Everything here is differentiable — scores, softmax and the weighted sum — so the whole system is trained with ordinary backpropagation. Nobody tells the model which source word corresponds to which target word; it learns to put its attention in useful places because that lowers the translation loss. This is why attention is called a *soft* alignment: instead of choosing exactly one source word (a hard decision that cannot be differentiated), it spreads a weight over all of them.

A worked example. Let there be three encoder states in two dimensions and a decoder state $s$:

$$h_1 = (1, 0), \quad h_2 = (0, 1), \quad h_3 = (1, 1), \quad s = (2, 0)$$

Use the simplest score, the dot product $\operatorname{score}(s, h_j) = s \cdot h_j$. Then $e = (2, 0, 2)$. With $e^2 \approx 7.389$ and $e^0 = 1$, the sum is $7.389 + 1 + 7.389 = 15.778$, so

$$\alpha = (0.468, \; 0.063, \; 0.468)$$

$$c_t = 0.468 \cdot (1, 0) + 0.063 \cdot (0, 1) + 0.468 \cdot (1, 1) = (0.937, \; 0.532)$$

The decoder state "points" in the direction of the first coordinate, so the two encoder states that have a 1 there receive almost all the attention, and $h_2$ is nearly ignored. The code below reproduces these numbers:

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max())               # subtract max for numerical stability
    return e / e.sum()

H = np.array([[1.0, 0.0],                  # encoder states h_1, h_2, h_3
              [0.0, 1.0],
              [1.0, 1.0]])
s = np.array([2.0, 0.0])                   # current decoder state

# Luong "dot" score: score(s, h_j) = s . h_j
scores = H @ s
alpha = softmax(scores)                    # attention weights, sum to 1
context = alpha @ H                        # weighted sum of encoder states
print("scores :", scores)
print("weights:", np.round(alpha, 3))
print("context:", np.round(context, 3))

# Bahdanau "additive" score: v . tanh(W s + U h_j)
rng = np.random.default_rng(0)
W, U = rng.normal(size=(3, 2)), rng.normal(size=(3, 2))
v = rng.normal(size=3)
add_scores = np.array([v @ np.tanh(W @ s + U @ h) for h in H])
print("additive weights:", np.round(softmax(add_scores), 3))
```

The second part of the code uses random parameters $W$, $U$, $v$, so its weights are close to uniform; in a trained model these parameters are learned.

## Bahdanau attention: the additive score

In the model of Bahdanau et al., the encoder is a bidirectional RNN (Chapter 11), so every $h_j$ contains information about the words both to the left and to the right of position $j$. The score is computed by a tiny feed-forward network with one hidden layer:

$$e_{tj} = v^\top \tanh\left(W s_{t-1} + U h_j\right)$$

Here $s_{t-1}$ is the previous decoder state, $W$ and $U$ are matrices that project the decoder and encoder states into a common space of some size $d_a$, and $v \in \mathbb{R}^{d_a}$ is a vector that turns the result into a single number. Because the two projected vectors are *added*, this is called **additive attention**. The resulting context $c_t$ is then fed into the RNN update that produces the new decoder state $s_t$, and finally the output word.

Additive attention is flexible — the small network can learn any kind of matching — but it requires a $\tanh$ and an extra matrix product for every pair $(t, j)$.

## Luong attention: multiplicative scores

Luong, Pham and Manning (2015) studied simpler variants. They compute attention from the *current* decoder state $s_t$ and proposed three score functions:

$$\operatorname{score}(s_t, h_j) = \begin{cases} s_t^\top h_j & \text{dot} \\ s_t^\top W_a\, h_j & \text{general} \\ v^\top \tanh\left(W_a [s_t; h_j]\right) & \text{concat} \end{cases}$$

The **dot** score is the plain dot product used in our worked example; it requires the decoder and encoder states to have the same size and has no parameters at all. The **general** score inserts a learned matrix $W_a$ between them. Both are forms of **multiplicative attention**, and they are cheap: for all source positions at once, the scores are a single matrix–vector product $H s_t$ (or $H W_a^\top s_t$), as in the line `scores = H @ s` above. The concat score is essentially Bahdanau's additive form.

After computing $c_t$, Luong's decoder combines it with $s_t$ into an "attentional" state

$$\tilde{s}_t = \tanh\left(W_c [c_t; s_t]\right)$$

and predicts the next word from $\tilde{s}_t$ with a softmax layer. Here $[c_t; s_t]$ is the concatenation of the two vectors. Luong et al. also described **local attention**, which looks only at a window of source positions around a predicted point, as a cheaper alternative to **global attention** over the whole sentence.

In practice the differences between these variants are modest. The dot-product family won in the long run because it maps perfectly onto fast matrix multiplication on GPUs. When we scale it properly in Chapter 15, it becomes the core of the Transformer.

## What attention weights show

Attention has a pleasant side effect: we can look inside the model. For each target word $t$ we have a distribution $\alpha_{t1}, \dots, \alpha_{tS}$ over source words. Putting these rows together gives a $T \times S$ matrix that can be drawn as a heat map. For English–French translation, Bahdanau et al. showed such maps: mostly diagonal, because the two languages often follow the same order, but with clear deviations where the order differs — for example, where an adjective comes after the noun in French. The model learned an **alignment** between the languages without ever being shown one.

For English and Korean the picture would be very different, since Korean usually places the verb at the end (subject–object–verb order). A plain seq2seq model would have to carry the English verb in its single vector across the whole sentence; with attention, the decoder simply looks back at the verb when it needs it.

Two cautions. First, attention weights are not a complete explanation of a model's decision; they show where information was read from, not why. We return to this in Chapter 40. Second, attention has a cost: for a source of length $S$ and a target of length $T$, we compute $S \cdot T$ scores. For sentences this is small; for very long documents it becomes important, as we will see in Part 4.

## From cross-attention to self-attention

It helps to describe attention with three roles, which will be central in the next chapter:

- the decoder state $s$ asks a question — it is the **query**;
- each encoder state $h_j$ is compared with the query — in this role it is a **key**;
- each encoder state $h_j$ also supplies the content that is mixed into the context — in this role it is a **value**.

In Bahdanau and Luong attention, the keys and the values are the same vectors $h_j$, and the query comes from a different sequence (the target). Attention between two different sequences is called **cross-attention**. A natural question follows: what if a sequence attends to *itself*? Each word could then gather information directly from every other word of the same sentence, in one step, without any recurrence. This is **self-attention**, the subject of Chapter 15.

## Summary

- Attention keeps all encoder states and builds a new context vector $c_t$ for every decoder step, removing the single-vector bottleneck.
- The computation has three steps: scores $e_j = \operatorname{score}(s, h_j)$, weights $\alpha = \operatorname{softmax}(e)$, context $c_t = \sum_j \alpha_j h_j$.
- Bahdanau (additive) attention scores with a small network, $v^\top \tanh(W s_{t-1} + U h_j)$; Luong (multiplicative) attention uses $s_t^\top h_j$ or $s_t^\top W_a h_j$.
- Attention is a soft, differentiable alignment learned from the translation loss alone; its weights can be visualized as a heat map.
- In the language of queries, keys and values, the decoder state is the query and the encoder states are keys and values; attention between two sequences is cross-attention.

## Exercises

1. Explain in your own words why attention solves the bottleneck problem of Chapter 13. Why is it important that the context vector is different at each decoder step?
2. Repeat the worked example with the decoder state $s = (0, 3)$. Compute the scores, the attention weights (to three decimals) and the context vector.
3. What happens to the attention weights in the worked example if we multiply $s$ by 10? By 0.1? Explain how the scale of the scores controls how "sharp" the attention is. (This observation will matter in Chapter 15.)
4. Modify the code so that it implements Luong's "general" score $s^\top W_a h_j$ with a random $2 \times 2$ matrix $W_a$, computing all scores with a single matrix product.

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| attention mechanism | механизм внимания | 어텐션 메커니즘 |
| attention weights | веса внимания | 어텐션 가중치 |
| attention score | оценка внимания | 어텐션 점수 |
| weighted sum | взвешенная сумма | 가중합 |
| alignment | выравнивание | 정렬 |
| soft alignment | мягкое выравнивание | 소프트 정렬 |
| additive attention | аддитивное внимание | 가산 어텐션 |
| multiplicative attention | мультипликативное внимание | 곱셈 어텐션 |
| dot product | скалярное произведение | 내적 |
| global attention | глобальное внимание | 전역 어텐션 |
| local attention | локальное внимание | 지역 어텐션 |
| cross-attention | перекрёстное внимание | 교차 어텐션 |
| query | запрос | 쿼리 |
| key | ключ | 키 |
| value | значение | 값 |

---
[← Previous: Sequence-to-Sequence and the Encoder–Decoder](13-sequence-to-sequence.md) · [Contents](README.md) · [Next: Self-Attention: Query, Key and Value →](15-self-attention.md)
