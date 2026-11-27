# Chapter 9. Backpropagation and Optimizers (SGD, Momentum, Adam)

> **Part 2 — Neural Network Foundations** · Level: Beginner
>
> **In this chapter you will learn:** how backpropagation computes every gradient with the chain rule · how PyTorch does this automatically · how the optimizers SGD, Momentum and Adam turn gradients into good updates

## The problem: millions of derivatives

In Chapter 8 we trained a line with two parameters, and we wrote the two derivatives by hand. A small language model has tens of millions of parameters, and a large one has billions. We need the derivative of the loss with respect to *every one* of them, at every training step.

One naive idea is the numerical derivative from Chapter 6: change one parameter by a tiny amount, run the network again, and see how the loss changes. But this needs at least one extra run of the whole network *per parameter*. For ten million parameters, that is ten million runs for a single step. Training would never finish.

**Backpropagation** solves this problem. It computes the exact gradient with respect to all parameters at a cost of roughly one extra pass through the network, about as expensive as computing the loss itself (a small constant factor, often quoted as two to three times one forward pass). The idea was popularized for neural networks by Rumelhart, Hinton and Williams in the 1980s, and it is still how every modern network is trained. The only mathematics it needs is the chain rule from Chapter 6.

## Computational graphs and the chain rule

Any computation of a network can be drawn as a **computational graph**: each node is a simple operation (multiply, add, ReLU, square), and arrows show which results feed into which. Computing the output from the input, following the arrows, is the **forward pass**.

Recall the chain rule: if $y$ depends on $u$, and $u$ depends on $x$, then

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}.$$

In words: the sensitivity of $y$ to $x$ is the sensitivity of $y$ to the intermediate $u$, times the sensitivity of $u$ to $x$. A long chain is just a longer product. If a variable influences the output through several paths, we add the contributions of all paths.

Backpropagation applies this rule systematically, from the end of the graph to the beginning. We start with $\frac{\partial L}{\partial L} = 1$ at the loss. Then, going backwards node by node, each node receives "the derivative of the loss with respect to my output", multiplies it by its own **local derivative** ("how does my output depend on my input?") and passes the result back to its inputs. This is the **backward pass**. Each node only needs to know its own simple operation; the chain rule connects them.

## A worked example by hand

Take a tiny network with one input, one hidden ReLU neuron and one output, without biases:

$$u = w_1 x, \qquad h = \operatorname{ReLU}(u), \qquad \hat{y} = w_2 h, \qquad L = (\hat{y} - y)^2.$$

Let $x = 2$, target $y = 5$, and current weights $w_1 = 1.5$, $w_2 = 2$.

**Forward pass.** $u = 1.5 \cdot 2 = 3$; $h = \operatorname{ReLU}(3) = 3$; $\hat{y} = 2 \cdot 3 = 6$; $L = (6 - 5)^2 = 1$.

**Backward pass.** We go from the loss back to the weights, keeping the values from the forward pass.

1. $\dfrac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y) = 2$.
2. Since $\hat{y} = w_2 h$: $\dfrac{\partial L}{\partial w_2} = \dfrac{\partial L}{\partial \hat{y}} \cdot h = 2 \cdot 3 = 6$, and $\dfrac{\partial L}{\partial h} = \dfrac{\partial L}{\partial \hat{y}} \cdot w_2 = 2 \cdot 2 = 4$.
3. ReLU has local derivative 1 when its input is positive (and 0 when negative). Here $u = 3 > 0$, so $\dfrac{\partial L}{\partial u} = 4 \cdot 1 = 4$.
4. Since $u = w_1 x$: $\dfrac{\partial L}{\partial w_1} = \dfrac{\partial L}{\partial u} \cdot x = 4 \cdot 2 = 8$.

So the gradient is $(\partial L / \partial w_1,\ \partial L / \partial w_2) = (8,\ 6)$. Both are positive, so gradient descent will *decrease* both weights, which lowers $\hat{y}$ from 6 towards the target 5. Notice two things. First, the backward pass reused values from the forward pass ($h$, $x$, $w_2$); this is why frameworks store intermediate results, and why training needs much more memory than just running a model. Second, the intermediate quantity $\partial L / \partial h$ was computed once and then used for everything before it. This reuse is what makes backpropagation efficient.

PyTorch's **automatic differentiation** (autograd) does exactly these steps for us:

```python
import torch

x, y = torch.tensor(2.0), torch.tensor(5.0)
w1 = torch.tensor(1.5, requires_grad=True)   # track gradients for w1
w2 = torch.tensor(2.0, requires_grad=True)   # and for w2

h = torch.relu(w1 * x)        # forward pass: h = 3
y_hat = w2 * h                # y_hat = 6
loss = (y_hat - y) ** 2       # loss = 1
loss.backward()               # backward pass fills .grad
print(w1.grad, w2.grad)       # tensor(8.) tensor(6.)
```

During the forward pass PyTorch records the graph; `loss.backward()` walks it backwards and stores each gradient in the `.grad` field of the parameter.

## Backpropagation through layers, and its dangers

For a whole layer $\mathbf{z} = W\mathbf{h} + \mathbf{b}$ the same rule works with vectors. Let $\boldsymbol{\delta} = \partial L / \partial \mathbf{z}$ be the gradient arriving from above. Then

$$\frac{\partial L}{\partial W} = \boldsymbol{\delta}\, \mathbf{h}^\top, \qquad \frac{\partial L}{\partial \mathbf{b}} = \boldsymbol{\delta}, \qquad \frac{\partial L}{\partial \mathbf{h}} = W^\top \boldsymbol{\delta}.$$

The first formula is an **outer product**: entry $(i, j)$ is $\delta_i h_j$, so a weight changes in proportion to its input and to the error signal of its output. The last formula sends the error to the layer below through the transposed matrix (Chapter 6). The backward pass therefore has the same structure as the forward pass, only in reverse.

In a deep network the gradient that reaches the first layers is a product of many local derivatives, one or more per layer. Products of many numbers are fragile. If the factors are mostly smaller than 1, the gradient shrinks exponentially with depth: this is the **vanishing gradient** problem. The derivative of the sigmoid is at most 0.25, so ten sigmoid layers can shrink the signal by a factor of about $0.25^{10} \approx 10^{-6}$. If the factors are mostly larger than 1, the gradient grows exponentially: the **exploding gradient** problem. These issues motivated ReLU (Chapter 7), careful weight initialization, normalization layers and residual connections (Chapter 18), and gradient clipping (Chapter 10). They are also the central difficulty of recurrent networks (Chapters 11 and 12).

## Optimizers: from gradients to updates

Backpropagation gives us the gradient $\mathbf{g} = \nabla_\theta L$. An **optimizer** decides how to use it to update the parameters. Three are essential.

**SGD.** The plain rule from Chapter 8, $\theta \leftarrow \theta - \eta\, \mathbf{g}$. In practice the gradient is computed on a small random subset of the data, a mini-batch (Chapter 10), so it is noisy; this is **stochastic gradient descent** (SGD). Its weakness shows in a long, narrow valley: the slope is steep across the valley and gentle along it. A learning rate large enough to move quickly along the valley makes the parameters jump from wall to wall across it, while a safe, small rate crawls.

**Momentum.** Imagine a heavy ball rolling down the valley. It keeps its velocity, so random sideways pushes cancel out and consistent pushes add up. We keep a running "velocity" $\mathbf{v}$:

$$\mathbf{v} \leftarrow \beta\, \mathbf{v} + \mathbf{g}, \qquad \theta \leftarrow \theta - \eta\, \mathbf{v}.$$

Here $\beta$ (typically 0.9) controls how much of the old velocity is kept. In the narrow valley, the gradients across the valley change sign every step and cancel, while the gradients along the valley point the same way and accumulate.

**Adam** (Kingma and Ba) combines momentum with a separate, adaptive step size for every parameter. It keeps two running averages, of the gradient and of the squared gradient:

$$\mathbf{m} \leftarrow \beta_1 \mathbf{m} + (1 - \beta_1)\, \mathbf{g}, \qquad \mathbf{v} \leftarrow \beta_2 \mathbf{v} + (1 - \beta_2)\, \mathbf{g}^2,$$

$$\hat{\mathbf{m}} = \frac{\mathbf{m}}{1 - \beta_1^t}, \qquad \hat{\mathbf{v}} = \frac{\mathbf{v}}{1 - \beta_2^t}, \qquad \theta \leftarrow \theta - \eta\, \frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{v}}} + \epsilon}.$$

The square $\mathbf{g}^2$, the square root and the division are taken component by component. $\mathbf{m}$ is the momentum (the "first moment"); $\mathbf{v}$ estimates the typical size of each gradient component (the "second moment"). Dividing by $\sqrt{\hat{\mathbf{v}}}$ gives each parameter a step of roughly size $\eta$, whether its gradients are large or small: steep directions are slowed down and flat directions are sped up. The hats are a **bias correction**: $\mathbf{m}$ and $\mathbf{v}$ start at zero, and dividing by $1 - \beta^t$ ($t$ is the step number) removes this early bias towards zero. $\epsilon$ is a tiny number such as $10^{-8}$ that prevents division by zero. Typical defaults are $\beta_1 = 0.9$, $\beta_2 = 0.999$. Adam, and especially its variant **AdamW** with decoupled weight decay (Chapter 10), is the standard optimizer for training Transformers and language models. Its price is memory: it stores two extra numbers for every parameter.

Let us compare the three on a narrow valley, $f(x, y) = x^2 + 20y^2$, starting from $(-4, 1)$, where $f = 36$:

```python
import torch

def f(p):                                  # a long, narrow valley:
    return p[0] ** 2 + 20 * p[1] ** 2      # flat along x, steep along y

def run(make_opt, steps=100):
    p = torch.tensor([-4.0, 1.0], requires_grad=True)
    opt = make_opt([p])
    for _ in range(steps):
        opt.zero_grad()                    # clear old gradients
        loss = f(p)
        loss.backward()                    # compute new gradients
        opt.step()                         # update the parameters
    return round(f(p).item(), 4)

print("SGD lr=0.01 ", run(lambda ps: torch.optim.SGD(ps, lr=0.01)))
print("SGD lr=0.05 ", run(lambda ps: torch.optim.SGD(ps, lr=0.05)))
print("Momentum    ", run(lambda ps: torch.optim.SGD(ps, lr=0.01, momentum=0.9)))
print("Adam        ", run(lambda ps: torch.optim.Adam(ps, lr=0.1)))
```

After 100 steps we get about 0.28 for SGD with `lr=0.01` (still crawling along the flat direction), exactly 20.0 for SGD with `lr=0.05` (it bounces between $y = 1$ and $y = -1$ forever, because along $y$ each step multiplies $y$ by $1 - 0.05 \cdot 40 = -1$), about 0.0008 for Momentum and about 0.0006 for Adam. The pattern `zero_grad()` → `backward()` → `step()` is the heart of every PyTorch training loop; we will use it in Chapter 10 and in every later chapter. Note that `zero_grad()` is needed because PyTorch *adds* new gradients to the old ones by default.

## Summary
- Backpropagation computes the gradient with respect to all parameters at the cost of about one extra pass, by applying the chain rule from the loss backwards.
- The forward pass computes and stores intermediate values; the backward pass multiplies local derivatives and reuses shared results.
- PyTorch autograd records the computational graph and fills `.grad` when we call `loss.backward()`.
- Long products of local derivatives can make gradients vanish or explode in deep networks.
- SGD steps against the gradient; Momentum accumulates a velocity; Adam adds a per-parameter adaptive step size and is the standard choice for Transformers.

## Exercises
1. Explain in your own words why backpropagation is much faster than computing a numerical derivative for every parameter.
2. Repeat the worked example with $w_1 = -1$ (other values unchanged). What is $h$? What are the two gradients? What does this tell you about "dead" ReLU neurons?
3. Add a bias $b$ to the output, $\hat{y} = w_2 h + b$, with $b = 0.5$. Compute $\partial L / \partial b$ by hand and check it with PyTorch.
4. In the optimizer code, try SGD with `lr=0.045` and Momentum with `momentum=0.5`. Explain the results using the idea of the narrow valley.

## Key terms
| English | Русский | 한국어 |
|---|---|---|
| backpropagation | метод обратного распространения ошибки | 역전파 |
| computational graph | вычислительный граф | 계산 그래프 |
| chain rule | цепное правило | 연쇄 법칙 |
| forward pass | прямой проход | 순전파 |
| backward pass | обратный проход | 역방향 패스 |
| local derivative | локальная производная | 국소 도함수 |
| automatic differentiation | автоматическое дифференцирование | 자동 미분 |
| vanishing gradient | затухающий градиент | 기울기 소실 |
| exploding gradient | взрывающийся градиент | 기울기 폭발 |
| optimizer | оптимизатор | 옵티마이저 |
| stochastic gradient descent | стохастический градиентный спуск | 확률적 경사 하강법 |
| momentum | импульс | 모멘텀 |
| bias correction | коррекция смещения | 편향 보정 |
| weight initialization | инициализация весов | 가중치 초기화 |

---
[← Previous: Loss Functions and Gradient Descent](08-loss-and-gradient-descent.md) · [Contents](README.md) · [Next: Training in Practice: Batches, Overfitting and Regularization →](10-training-in-practice.md)
