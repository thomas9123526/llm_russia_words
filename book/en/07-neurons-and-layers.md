# Chapter 7. Neurons, Layers and Activation Functions

> **Part 2 — Neural Network Foundations** · Level: Beginner
>
> **In this chapter you will learn:** what an artificial neuron computes · why a network needs non-linear activation functions · how layers of neurons form a multilayer perceptron

## A small puzzle: XOR

Consider four inputs, each made of two bits, and a rule called XOR ("exclusive or"): the answer is 1 when exactly one of the bits is 1, and 0 otherwise.

| $x_1$ | $x_2$ | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Draw the four points on a plane. The two points with answer 1 sit on one diagonal, the two points with answer 0 on the other. Can you draw *one straight line* that separates them? Try it: it is impossible. This tiny puzzle explains why neural networks have several layers and why they need non-linear functions. In this chapter we build a network that solves XOR, and on the way we meet the basic parts of every modern language model.

## The artificial neuron

An **artificial neuron** is a very simple function. It takes an input vector $\mathbf{x} = (x_1, \dots, x_n)$, computes a weighted sum, adds a number, and passes the result through a function:

$$z = \mathbf{w} \cdot \mathbf{x} + b = \sum_{i=1}^{n} w_i x_i + b, \qquad a = \sigma(z).$$

The symbols are:
- $w_i$ are the **weights**. Each weight says how important input $x_i$ is, and whether it pushes the output up (positive weight) or down (negative weight).
- $b$ is the **bias**. It shifts the result and lets the neuron produce a non-zero output even when all inputs are zero.
- $z$ is the **pre-activation**, the raw weighted sum.
- $\sigma$ is the **activation function**, and $a$ is the neuron's output, its **activation**.

The weights and the bias are the **parameters** of the neuron. They are not chosen by hand; training (Chapters 8 and 9) finds them from data. The name "neuron" comes from a loose analogy with brain cells, which collect signals and "fire" when the total is strong enough. The analogy should not be taken too far. Mathematically, a neuron is a dot product (Chapter 6) followed by a simple function.

Geometrically, the equation $\mathbf{w} \cdot \mathbf{x} + b = 0$ describes a straight line in two dimensions (a plane or hyperplane in more). Points on one side give $z > 0$, points on the other side give $z < 0$. So a single neuron is a **linear classifier**: it can only separate classes with a straight boundary. If the activation is the sigmoid and we read the output as a probability, a single neuron is exactly **logistic regression**. And this is why one neuron cannot solve XOR: no straight line separates the two diagonals.

## Why we need non-linearity

The natural idea is to use several neurons and to feed their outputs into further neurons. A group of neurons that read the same input forms a **layer**. If a layer has $m$ neurons, we stack their weight vectors as the rows of a matrix $W \in \mathbb{R}^{m \times n}$ and their biases into a vector $\mathbf{b} \in \mathbb{R}^m$. Then the whole layer is one line:

$$\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b}),$$

where $\sigma$ is applied to each component separately. This is the matrix–vector product from Chapter 6, plus a bias, plus an activation. Such a layer, where every output is connected to every input, is called a **fully connected layer** (in PyTorch, `nn.Linear` followed by an activation).

Now suppose we forget the activation and simply stack two linear layers:

$$W_2(W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2W_1)\mathbf{x} + (W_2\mathbf{b}_1 + \mathbf{b}_2).$$

The result is again one matrix times $\mathbf{x}$ plus one vector. In other words, two linear layers are just one linear layer in disguise. A hundred linear layers would still draw only straight boundaries, and still fail on XOR. The activation function breaks this collapse. Because $\sigma$ bends its input, the composition of layers can bend the input space, and the network can represent curved boundaries and complicated functions. Depth is useful *only* because of non-linearity.

## A tour of activation functions

Many activation functions have been proposed. Four are enough for this book.

**Sigmoid.** $\sigma(z) = \dfrac{1}{1 + e^{-z}}$. It squashes any number into the interval $(0, 1)$, so it is natural for probabilities of a yes/no answer. Its weakness: for large $|z|$ the curve is almost flat. We say the neuron **saturates**, and its derivative is nearly zero. In Chapter 9 we will see that tiny derivatives slow learning in deep networks.

**Hyperbolic tangent.** $\tanh(z) = \dfrac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$. It has the same S-shape as the sigmoid but maps into $(-1, 1)$ and is centred at zero. It also saturates. We will meet it again inside recurrent networks and LSTMs (Chapters 11 and 12).

**ReLU** (rectified linear unit). $\operatorname{ReLU}(z) = \max(0, z)$. It keeps positive values and replaces negative ones with zero. It is extremely cheap to compute, and for positive inputs its derivative is exactly 1, so it does not saturate there. ReLU made it much easier to train deep networks and became the default choice for many years. Its weakness: a neuron whose input is always negative outputs zero and gets zero gradient, so it may stop learning (a "dead" neuron).

**GELU** (Gaussian error linear unit). $\operatorname{GELU}(z) = z \cdot \Phi(z)$, where $\Phi$ is the cumulative distribution function of the standard normal distribution. It behaves like a smooth version of ReLU: close to $z$ for large positive inputs, close to 0 for large negative inputs, and smooth around zero, with a small negative dip. GELU is used in BERT and GPT models (Chapters 22 and 23), and in Chapter 18 we will meet its relatives SwiGLU and GLU.

Here are the four functions at a few points (computed by the code at the end of the chapter):

| $z$ | −2 | −0.5 | 0 | 0.5 | 2 |
|---|---|---|---|---|---|
| sigmoid | 0.119 | 0.378 | 0.5 | 0.622 | 0.881 |
| tanh | −0.964 | −0.462 | 0 | 0.462 | 0.964 |
| ReLU | 0 | 0 | 0 | 0.5 | 2 |
| GELU | −0.046 | −0.154 | 0 | 0.346 | 1.954 |

## Layers make a network: the multilayer perceptron

When we chain several layers, we get a **multilayer perceptron** (MLP), also called a **feed-forward network**: information flows in one direction, from input to output. The layers between input and output are **hidden layers**, because we never observe their values directly in the data. A network with two hidden layers computes

$$\mathbf{h}_1 = \sigma(W_1\mathbf{x} + \mathbf{b}_1), \quad \mathbf{h}_2 = \sigma(W_2\mathbf{h}_1 + \mathbf{b}_2), \quad \mathbf{y} = W_3\mathbf{h}_2 + \mathbf{b}_3.$$

The last layer usually has no activation, or has one that suits the task: for classification it outputs logits, which softmax turns into probabilities (Chapter 6). The number of neurons in a layer is its **width**; the number of layers is the **depth**. A layer from dimension $n$ to dimension $m$ has $m \cdot n$ weights and $m$ biases, so $mn + m$ parameters.

How powerful is an MLP? The **universal approximation theorem** says that an MLP with a single hidden layer and a suitable non-linear activation can approximate any continuous function on a bounded region as closely as we like, if the hidden layer is wide enough. This is reassuring, but it says nothing about how many neurons are needed or how to find the weights. In practice, deeper networks usually reach the same quality with far fewer parameters, because each layer can build on features computed by the layer before.

Now let us solve XOR with a 2–2–1 network: two inputs, two hidden ReLU neurons, one output. We set the weights by hand:

$$h_1 = \operatorname{ReLU}(x_1 + x_2), \qquad h_2 = \operatorname{ReLU}(x_1 + x_2 - 1), \qquad y = h_1 - 2h_2.$$

Check all four inputs. For $(0,0)$: $h = (0, 0)$, $y = 0$. For $(0,1)$ and $(1,0)$: $h = (1, 0)$, $y = 1$. For $(1,1)$: $h = (2, 1)$, $y = 2 - 2 = 0$. It works. Look at what happened: the hidden layer moved the points into a new space where the two "1" inputs land on the same point, $(1, 0)$, and a straight line *does* separate the classes. This is the central idea of deep learning: hidden layers learn a new **representation** of the input in which the task becomes easy. Word embeddings from Chapter 5 are one example of such a learned representation.

```python
import torch
import torch.nn as nn

# The four inputs of the XOR problem
x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])

# Hand-set weights for a 2-2-1 network that computes XOR
W1 = torch.tensor([[1., 1.],     # hidden neuron 1
                   [1., 1.]])    # hidden neuron 2
b1 = torch.tensor([0., -1.])
w2 = torch.tensor([1., -2.])

h = torch.relu(x @ W1.T + b1)    # hidden layer, shape (4, 2)
y = h @ w2                       # output, shape (4,)
print(y)                         # tensor([0., 1., 1., 0.])

# Common activation functions at a few points
z = torch.tensor([-2., -0.5, 0., 0.5, 2.])
for name, fn in [("sigmoid", torch.sigmoid), ("tanh", torch.tanh),
                 ("relu", torch.relu), ("gelu", nn.functional.gelu)]:
    print(f"{name:8s}", fn(z).numpy().round(3))

# A network of the same shape, built from PyTorch modules
mlp = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
print(mlp(x).shape)                               # torch.Size([4, 1])
print(sum(p.numel() for p in mlp.parameters()))   # 9
```

Note that `x @ W1.T` processes all four examples at once: `x` has shape (4, 2), one example per row, exactly as described in Chapter 6. The `nn.Sequential` model starts with random weights, so its outputs are meaningless for now. In the next two chapters we will teach it to find good weights by itself.

## Summary
- An artificial neuron computes a weighted sum of its inputs plus a bias, $z = \mathbf{w} \cdot \mathbf{x} + b$, and applies an activation function.
- A single neuron draws a straight boundary, so it cannot solve problems such as XOR.
- Without non-linear activations, any stack of linear layers collapses into one linear layer.
- Common activations are sigmoid, tanh, ReLU and GELU; sigmoid and tanh saturate, ReLU is cheap and does not saturate for positive inputs, GELU is a smooth ReLU used in Transformers.
- A multilayer perceptron chains fully connected layers; hidden layers learn new representations in which the task becomes easier.

## Exercises
1. Explain in your own words why two linear layers without an activation are no more powerful than one.
2. A neuron has weights $\mathbf{w} = (2, -1)$, bias $b = 0.5$ and ReLU activation. Compute its output for $\mathbf{x} = (1, 3)$ and for $\mathbf{x} = (1, 1)$.
3. Count the parameters of an MLP with layer sizes 784 → 256 → 128 → 10. Check your answer with `nn.Sequential` and `p.numel()`.
4. Find hand-set weights for a single neuron with a step activation ($1$ if $z > 0$, else $0$) that computes AND of two bits. Why is this possible for AND but not for XOR?

## Key terms
| English | Русский | 한국어 |
|---|---|---|
| artificial neuron | искусственный нейрон | 인공 뉴런 |
| weight | вес | 가중치 |
| bias | смещение | 편향 |
| activation function | функция активации | 활성화 함수 |
| parameter | параметр | 매개변수(파라미터) |
| linear classifier | линейный классификатор | 선형 분류기 |
| logistic regression | логистическая регрессия | 로지스틱 회귀 |
| layer | слой | 층 |
| fully connected layer | полносвязный слой | 완전 연결 층 |
| sigmoid | сигмоида | 시그모이드 |
| saturation | насыщение | 포화 |
| multilayer perceptron | многослойный перцептрон | 다층 퍼셉트론 |
| hidden layer | скрытый слой | 은닉층 |
| universal approximation theorem | универсальная теорема аппроксимации | 보편 근사 정리 |
| representation | представление | 표현 |

---
[← Previous: The Math You Need: Vectors, Matrices, Probability](06-math-you-need.md) · [Contents](README.md) · [Next: Loss Functions and Gradient Descent →](08-loss-and-gradient-descent.md)
