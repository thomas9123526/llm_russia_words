# Chapter 8. Loss Functions and Gradient Descent

> **Part 2 — Neural Network Foundations** · Level: Beginner
>
> **In this chapter you will learn:** how a loss function turns "how wrong is the model?" into one number · why cross-entropy is the loss of language models · how gradient descent uses derivatives to reduce the loss step by step

## How does a model know it is wrong?

At the end of Chapter 7 we built a network with random weights. Its outputs were meaningless. Suppose we show it the sentence "The cat sat on the" and it gives the next word *mat* a probability of only 0.01. We know this is bad. But a computer needs a precise answer to two questions: *how bad* is it, and *which way* should each weight move to make it better?

The answer to the first question is a **loss function**: a function that takes the model's prediction and the correct answer and returns one non-negative number, large when the prediction is poor and small when it is good. The answer to the second question is **gradient descent**: compute the derivative of the loss with respect to every weight (Chapter 6) and move each weight a little in the direction that lowers the loss. These two ideas are the engine of all neural network training, from a two-neuron toy to a large language model.

## Loss functions: measuring error with one number

We collect all parameters of the model (weights and biases) into one vector $\theta$ (the Greek letter "theta"). The model with parameters $\theta$ maps an input $x$ to a prediction $\hat{y} = f_\theta(x)$. A training set contains $N$ pairs $(x_i, y_i)$ of inputs and correct answers, called **targets** or **labels**. The loss on the whole set is the average loss per example:

$$L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \ell(f_\theta(x_i),\ y_i),$$

where $\ell$ is the loss of a single example. Training means finding $\theta$ that makes $L(\theta)$ small.

**Mean squared error.** When the target is a real number (predicting a price, a temperature, a score), the usual choice is the **mean squared error** (MSE):

$$L_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2.$$

The difference $\hat{y}_i - y_i$ is the error on example $i$. Squaring makes every error positive and punishes large errors much more than small ones: an error of 2 costs four times as much as an error of 1.

**Cross-entropy.** When the target is a class, such as the next token from a vocabulary, the model outputs a probability distribution $p$ over the $K$ classes (via softmax, Chapter 6). Let $c$ be the index of the correct class. The **cross-entropy loss** is

$$\ell_{\text{CE}} = -\log p_c.$$

Only the probability of the correct class matters. If the model gives it probability 1, the loss is $-\log 1 = 0$. If the probability is small, the loss is large: $-\log 0.5 \approx 0.69$, $-\log 0.01 \approx 4.6$. Take our logits $(2.0,\ 1.0,\ 0.1)$ from Chapter 6, with softmax $(0.659,\ 0.242,\ 0.099)$. If the correct class is the first one, the loss is $-\log 0.659 \approx 0.417$. If it is the third one, the loss is $-\log 0.099 \approx 2.317$. The model is "surprised" by an answer it considered unlikely, and the loss measures this surprise.

Why exactly this formula? It comes from a principle called **maximum likelihood**. A good model should assign high probability to the data we actually observed. The probability of the whole data set is a product of per-example probabilities. Taking the logarithm (Chapter 6) turns the product into a sum, and putting a minus sign turns "maximize" into "minimize". So minimizing average cross-entropy is the same as maximizing the likelihood of the data. Cross-entropy is therefore also called the **negative log-likelihood**.

For a language model, every position in the text is one classification problem: predict token $w_t$ from the previous tokens. The training loss is the average over all positions:

$$L = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(w_t \mid w_1, \dots, w_{t-1}).$$

This is the same quantity we met in Chapter 4: **perplexity** is simply $e^{L}$. A model with loss 2.0 has perplexity $e^{2} \approx 7.4$; it is as uncertain as if it chose uniformly among about seven tokens at every step. Reducing the loss and reducing perplexity are the same goal.

## Gradient descent: walking downhill

Imagine the loss as a landscape. Each point on the ground is one choice of parameters $\theta$, and the height is $L(\theta)$. We stand somewhere in fog and can only feel the slope under our feet. To go down, we step in the direction of steepest descent. In Chapter 6 we learned that the gradient $\nabla L$ points in the direction of steepest *ascent*. So we step against it:

$$\theta \leftarrow \theta - \eta\, \nabla_\theta L(\theta).$$

Here $\leftarrow$ means "replace by", $\nabla_\theta L$ is the gradient of the loss with respect to the parameters, and $\eta$ (the Greek letter "eta") is the **learning rate**, a small positive number that controls the step size. One such update is called a **step** or **iteration**. We repeat it many times.

A worked example with one parameter. Let $L(w) = (w - 3)^2$, whose minimum is at $w = 3$. The derivative is $L'(w) = 2(w - 3)$. Start at $w = 0$ with $\eta = 0.1$:

| step | $w$ | $L(w)$ | $L'(w)$ |
|---|---|---|---|
| 0 | 0 | 9 | −6 |
| 1 | 0.6 | 5.76 | −4.8 |
| 2 | 1.08 | 3.69 | −3.84 |
| 3 | 1.464 | 2.36 | −3.07 |
| 5 | 2.017 | 0.97 | −1.97 |

Each step moves $w$ towards 3, and the steps become smaller as the slope flattens. In fact, here $w - 3$ is multiplied by $1 - 2\eta = 0.8$ at every step, so after $t$ steps $w_t = 3 - 3 \cdot 0.8^t$.

Real networks are not this simple. Their loss landscapes are **non-convex**: they have many valleys, flat regions and saddle points, and gradient descent finds *a* good low point, not necessarily the lowest. Surprisingly, for large neural networks this usually works very well in practice.

## The learning rate: not too small, not too large

The learning rate is the most important **hyperparameter**, a setting that we choose rather than learn. In our example, $w - 3$ is multiplied by $(1 - 2\eta)$ each step. Look at what different values do:

- $\eta = 0.01$: the factor is 0.98. Training converges, but very slowly.
- $\eta = 0.1$: the factor is 0.8. Fast, smooth convergence.
- $\eta = 0.9$: the factor is $-0.8$. $w$ jumps over the minimum to the other side each step, but still converges.
- $\eta = 1.1$: the factor is $-1.2$. Every jump is larger than the previous one, and training **diverges**: the loss grows without limit.

In a real network the same picture holds in many dimensions at once. Too small a learning rate wastes compute; too large a rate makes the loss jump or explode. A common practical sign of a too-large rate is a loss that suddenly becomes `nan`. Chapter 9 introduces optimizers such as Adam that adapt the step size, and Chapter 10 shows learning rate schedules that change $\eta$ during training.

One more useful fact. For softmax followed by cross-entropy, the gradient with respect to the logits has a beautifully simple form:

$$\frac{\partial \ell_{\text{CE}}}{\partial z_k} = p_k - y_k,$$

where $y_k$ is 1 for the correct class and 0 otherwise (a **one-hot** vector). For our example, if the first class is correct, the gradient is $(0.659 - 1,\ 0.242,\ 0.099) = (-0.341,\ 0.242,\ 0.099)$. Gradient descent therefore raises the logit of the correct class and lowers the others, each in proportion to how wrong it was.

## Gradient descent in code

Let us fit a straight line $\hat{y} = wx + b$ to noisy points generated from $y = 2x + 1$. The loss is MSE. Its partial derivatives are

$$\frac{\partial L}{\partial w} = \frac{2}{N} \sum_i (\hat{y}_i - y_i)\, x_i, \qquad \frac{\partial L}{\partial b} = \frac{2}{N} \sum_i (\hat{y}_i - y_i).$$

```python
import numpy as np

# Toy data: y = 2x + 1 plus a little noise
rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, size=100)
y = 2.0 * x + 1.0 + 0.1 * rng.normal(size=100)

w, b = 0.0, 0.0                  # start from zero
lr = 0.1                         # learning rate
for step in range(201):
    err = (w * x + b) - y        # prediction minus target
    loss = np.mean(err ** 2)     # mean squared error
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss:.4f}  w {w:.3f}  b {b:.3f}")
    grad_w = 2 * np.mean(err * x)    # dL/dw
    grad_b = 2 * np.mean(err)        # dL/db
    w -= lr * grad_w                 # step against the gradient
    b -= lr * grad_b
```

The output shows the loss falling from about 2.87 to about 0.009, and the parameters approaching $w \approx 1.99$, $b \approx 0.99$. They do not reach exactly 2 and 1, because the data contain noise; the loss cannot go below the noise level (about $0.1^2 = 0.01$).

In PyTorch, cross-entropy is a single function. It takes raw logits, not probabilities, and applies softmax internally in a numerically stable way:

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 1.0, 0.1]])            # one example, three classes
print(F.cross_entropy(logits, torch.tensor([0])))   # correct class 0: 0.417
print(F.cross_entropy(logits, torch.tensor([2])))   # correct class 2: 2.317
```

Here we computed the gradients by hand. For a network with millions of parameters and many layers this is impossible. In Chapter 9 we will learn backpropagation, the algorithm that computes all these derivatives automatically.

## Summary
- A loss function turns the quality of a prediction into one number; training minimizes its average over the data.
- Mean squared error is used for real-valued targets; cross-entropy, $-\log p_c$, is used for classes.
- Minimizing cross-entropy is maximum likelihood; for language models, perplexity equals $e^{L}$.
- Gradient descent repeats $\theta \leftarrow \theta - \eta \nabla L$, moving the parameters against the gradient.
- The learning rate controls the step size: too small is slow, too large oscillates or diverges.
- For softmax plus cross-entropy, the gradient with respect to the logits is $p - y$.

## Exercises
1. Explain in your own words why the cross-entropy loss depends only on the probability of the correct class, and what happens to the loss when that probability approaches zero.
2. A language model has average cross-entropy 3.0 on a test text. What is its perplexity? What loss corresponds to perplexity 20?
3. For $L(w) = (w - 3)^2$, start at $w = 0$ and do three steps with $\eta = 0.9$ by hand. Then do three steps with $\eta = 1.1$. Describe what you see.
4. In the code above, try `lr = 0.01` and `lr = 1.2`. How many steps are needed in the first case to reach loss below 0.02? What happens in the second case, and why?

## Key terms
| English | Русский | 한국어 |
|---|---|---|
| loss function | функция потерь | 손실 함수 |
| target (label) | целевое значение (метка) | 목표값(레이블) |
| mean squared error | среднеквадратичная ошибка | 평균 제곱 오차 |
| cross-entropy | перекрёстная энтропия | 교차 엔트로피 |
| maximum likelihood | максимальное правдоподобие | 최대 우도 |
| negative log-likelihood | отрицательный логарифм правдоподобия | 음의 로그 우도 |
| perplexity | перплексия | 퍼플렉서티 |
| gradient descent | градиентный спуск | 경사 하강법 |
| learning rate | скорость обучения | 학습률 |
| iteration (step) | итерация (шаг) | 반복(스텝) |
| hyperparameter | гиперпараметр | 하이퍼파라미터 |
| convergence | сходимость | 수렴 |
| divergence | расходимость | 발산 |
| one-hot vector | унитарный вектор (one-hot) | 원-핫 벡터 |

---
[← Previous: Neurons, Layers and Activation Functions](07-neurons-and-layers.md) · [Contents](README.md) · [Next: Backpropagation and Optimizers (SGD, Momentum, Adam) →](09-backpropagation-and-optimizers.md)
