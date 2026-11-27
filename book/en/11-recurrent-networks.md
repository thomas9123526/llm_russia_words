# Chapter 11. Recurrent Neural Networks

> **Part 3 — Sequences and Attention** · Level: Intermediate
>
> **In this chapter you will learn:** how a network can read a sequence one step at a time · what a hidden state is and how weights are shared across time · why gradients vanish or explode in long sequences

## A network that remembers

Consider two short sentences:

- "The dog bit the man."
- "The man bit the dog."

They contain exactly the same words. A bag-of-words model from Chapter 4 sees them as identical, and so does a simple feed-forward network that receives the average of the word embeddings from Chapter 5. Yet the meanings are opposite. The difference lies entirely in the **order** of the words.

The networks we built in Part 2 take one fixed-size vector as input and produce one output. Language does not fit this shape well. Sentences have different lengths, and the meaning of a word depends on what came before it. We could concatenate all embeddings into one long vector, but then a 7-word sentence and a 12-word sentence would need different input sizes, and the network would have to learn separately what "dog" means in position 2 and in position 5.

A **recurrent neural network** (RNN) solves this in a natural way. It reads the sequence like a person reads a sentence: one token at a time, from left to right. After each token it updates a small internal summary called the **hidden state**. The hidden state is the network's memory: it carries information about everything it has read so far. When the next token arrives, the network combines it with this memory and produces a new memory.

The key idea is that the same small network is used at every step. It does not matter whether the sentence has 3 tokens or 300 — we just apply the same operation again and again. This is where the word "recurrent" comes from: the output of the step is fed back as an input to the next step.

## The recurrence: one formula, many steps

Let the input sequence be $x_1, x_2, \dots, x_T$, where each $x_t$ is a vector — usually the embedding of the token at position $t$ (Chapter 5). The index $t$ is called the **time step**, even though nothing here is about physical time; it simply means "position in the sequence".

The RNN keeps a hidden state $h_t$, a vector of size $d_h$. We start with $h_0 = 0$, and at every step we compute

$$h_t = \tanh\left(W_{xh}\, x_t + W_{hh}\, h_{t-1} + b_h\right)$$

Here:

- $x_t \in \mathbb{R}^{d_{in}}$ is the input at step $t$;
- $h_{t-1} \in \mathbb{R}^{d_h}$ is the previous hidden state;
- $W_{xh}$ is a $d_h \times d_{in}$ matrix that transforms the input;
- $W_{hh}$ is a $d_h \times d_h$ matrix that transforms the old memory;
- $b_h$ is a bias vector;
- $\tanh$ is the hyperbolic tangent activation (Chapter 7), which keeps every component between $-1$ and $1$.

If we need a prediction at step $t$ — for example, a probability distribution over the next token — we add an output layer:

$$y_t = \operatorname{softmax}\left(W_{hy}\, h_t + b_y\right)$$

This is exactly a small neural network from Chapter 7 with one extra input: its own previous state.

Notice that $W_{xh}$, $W_{hh}$ and $b_h$ have no index $t$. The same parameters are used at every step. This is called **weight sharing** across time. It has two important consequences. First, the number of parameters does not depend on the sequence length. Second, whatever the network learns about a pattern (say, "an adjective is often followed by a noun") it can apply at any position.

Here is the forward pass in NumPy:

```python
import numpy as np

rng = np.random.default_rng(0)
d_in, d_h, T = 3, 4, 5                   # input size, hidden size, sequence length

W_xh = rng.normal(0, 0.5, (d_h, d_in))   # input -> hidden
W_hh = rng.normal(0, 0.5, (d_h, d_h))    # hidden -> hidden (the "memory" weights)
b_h = np.zeros(d_h)

xs = rng.normal(size=(T, d_in))          # a sequence of 5 input vectors
h = np.zeros(d_h)                        # initial hidden state h_0
for t, x in enumerate(xs, start=1):
    h = np.tanh(W_xh @ x + W_hh @ h + b_h)   # the same weights at every step
    print(f"h_{t} =", np.round(h, 3))

# How a gradient shrinks when it flows back through many steps:
# multiply by a factor like 0.5 (a small derivative) again and again
for steps in [1, 5, 10, 20, 50]:
    print(steps, "steps back:", 0.5 ** steps)
```

The loop is the whole RNN. The final $h_5$ depends on all five inputs, because each step passes information forward through $h$.

## Unrolling the network and training it

It is helpful to draw the RNN **unrolled** in time. Instead of one box with a loop, we draw one copy of the box for each time step, connected left to right:

```
 x_1        x_2        x_3
  |          |          |
[RNN] -h_1-> [RNN] -h_2-> [RNN] -h_3-> ...
  |          |          |
 y_1        y_2        y_3
```

All copies share the same weights. Once the network is unrolled, it is just a very deep feed-forward network, with one "layer" per time step. So we can train it with the tools from Chapters 8 and 9: compute a loss at each step (for language modeling, the cross-entropy of the next token), add the losses together, and run backpropagation through the unrolled graph. This procedure is called **backpropagation through time** (BPTT). Because the weights are shared, the gradient for $W_{hh}$ is the sum of its gradients from all time steps.

For very long texts, unrolling thousands of steps is too expensive in memory. In practice we use **truncated BPTT**: we cut the text into chunks of, say, 100 tokens, carry the hidden state forward from one chunk to the next, but backpropagate only inside each chunk.

RNNs can be used in several configurations:

- **many-to-one**: read a whole sentence and use the last hidden state $h_T$ to classify it (for example, sentiment analysis);
- **many-to-many (aligned)**: produce an output at every step, as in language modeling or part-of-speech tagging;
- **many-to-many (not aligned)**: read one sequence and produce another of a different length, as in machine translation — the subject of Chapter 13.

Two common extensions are worth knowing. A **bidirectional RNN** runs one RNN left to right and another right to left, and concatenates their hidden states, so that each position sees both its left and right context. A **stacked RNN** puts several RNN layers on top of each other: the hidden states of layer 1 become the inputs of layer 2.

## Vanishing and exploding gradients

In theory an RNN can remember information from very far back. In practice, simple RNNs struggle to learn dependencies longer than a few dozen steps. Consider the sentence: "The keys that I left on the table in the kitchen yesterday *are* gone." To choose "are" rather than "is", the model must remember the word "keys" across many intermediate tokens. This is a **long-range dependency**.

The problem appears during training. To learn that "keys" influences "are", the gradient of the loss at the step of "are" must travel back to the step of "keys". By the chain rule (Chapter 9), the gradient that flows from $h_t$ back to $h_k$ is a product of many Jacobian matrices:

$$\frac{\partial h_t}{\partial h_k} = \prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}} = \prod_{j=k+1}^{t} \operatorname{diag}\!\left(1 - h_j^2\right) W_{hh}$$

Here $\operatorname{diag}(1 - h_j^2)$ is the derivative of $\tanh$, written as a diagonal matrix, and the product has $t - k$ factors. Think of the one-dimensional case: we multiply the same kind of number again and again. If the typical factor is smaller than 1, the product shrinks exponentially — the last lines of the code above show that $0.5^{20}$ is already about $10^{-6}$. This is the **vanishing gradient** problem: the signal from distant tokens becomes so small that the weights hardly learn from it. If the typical factor is larger than 1, the product grows exponentially, and we get the **exploding gradient** problem: huge updates that destroy training and may produce `NaN` values.

Since the derivative of $\tanh$ is at most 1 and is often much smaller, vanishing is the more common case. The analysis was described clearly by Bengio, Simard and Frasconi (1994) and later by Pascanu, Mikolov and Bengio (2013).

Exploding gradients have a simple practical fix: **gradient clipping**. Before each update, we compute the norm of the whole gradient vector $g$; if it is larger than a threshold $\tau$, we rescale it:

$$g \leftarrow g \cdot \frac{\tau}{\lVert g \rVert} \quad \text{if } \lVert g \rVert > \tau$$

The direction of the step stays the same; only its length is limited. Vanishing gradients are harder. Careful initialization helps a little, but the real solution is to change the architecture so that information can flow across many steps without being multiplied by a small factor each time. This is exactly what the LSTM and GRU do, as we will see in Chapter 12.

## Strengths and limits of RNNs

RNNs were the main tool for language modeling, speech recognition and machine translation for many years. Their strengths are clear: they handle sequences of any length, they share parameters across positions, and at generation time each new step costs the same, because the whole past is compressed into one fixed-size vector $h_t$.

The same properties are also their weaknesses. First, a fixed-size hidden state is a bottleneck: everything the model knows about a long text must fit into a few hundred numbers. Second, computation is **sequential**: $h_t$ cannot be computed before $h_{t-1}$, so we cannot process the tokens of a sentence in parallel on a GPU. Third, even with clipping, plain RNNs remember long-range information poorly.

The rest of Part 3 is a story of fixing these problems one by one. LSTM and GRU (Chapter 12) fix the memory. Encoder–decoder models (Chapter 13) apply RNNs to translation and reveal the bottleneck. Attention (Chapter 14) removes the bottleneck, and self-attention (Chapter 15) finally removes the recurrence itself, which leads to the Transformer in Part 4.

## Summary

- An RNN reads a sequence one step at a time and keeps a hidden state $h_t$ that summarizes everything read so far.
- The update $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$ uses the same weights at every time step (weight sharing).
- Training unrolls the network in time and applies backpropagation through time (BPTT), often truncated to chunks.
- Gradients that flow across many steps are products of many factors, so they tend to vanish or explode.
- Gradient clipping fixes exploding gradients; vanishing gradients require new architectures such as LSTM and GRU.
- RNNs are sequential and compress the past into a fixed-size vector, which limits speed and long-range memory.

## Exercises

1. Explain in your own words why the sentences "The dog bit the man" and "The man bit the dog" get the same representation in a bag-of-words model but can get different final hidden states in an RNN.
2. An RNN has input size $d_{in} = 50$ and hidden size $d_h = 128$. How many parameters do $W_{xh}$, $W_{hh}$ and $b_h$ contain together? Does this number change if the sentence has 10 or 1,000 tokens?
3. In the one-dimensional case, suppose each backward step multiplies the gradient by 0.9. After how many steps does the gradient fall below 1% of its original size? What if the factor is 1.1 — after how many steps does it become 100 times larger?
4. Modify the code above so that it stores all hidden states in a $T \times d_h$ array, and then implement gradient clipping as a function `clip(g, tau)` that takes a NumPy vector and returns the rescaled vector.

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| recurrent neural network (RNN) | рекуррентная нейронная сеть | 순환 신경망 |
| hidden state | скрытое состояние | 은닉 상태 |
| time step | временной шаг | 타임 스텝 |
| weight sharing | совместное использование весов | 가중치 공유 |
| unrolling | развёртка во времени | 펼치기(언롤링) |
| backpropagation through time (BPTT) | обратное распространение ошибки во времени | 시간에 따른 역전파 |
| truncated BPTT | усечённое обратное распространение во времени | 절단된 BPTT |
| long-range dependency | долгосрочная зависимость | 장거리 의존성 |
| vanishing gradient | затухающий градиент | 기울기 소실 |
| exploding gradient | взрывающийся градиент | 기울기 폭발 |
| gradient clipping | обрезка градиентов | 그래디언트 클리핑 |
| bidirectional RNN | двунаправленная рекуррентная сеть | 양방향 RNN |
| stacked RNN | многослойная рекуррентная сеть | 다층 RNN |

---
[← Previous: Training in Practice: Batches, Overfitting and Regularization](10-training-in-practice.md) · [Contents](README.md) · [Next: LSTM and GRU →](12-lstm-and-gru.md)
