# 제9장. 역전파와 옵티마이저 (SGD, 모멘텀, Adam)

> **제2부 — 신경망의 기초** · 수준: 초급
>
> **이 장에서 배울 내용:** 역전파가 연쇄 법칙으로 모든 그래디언트를 계산하는 방법 · PyTorch가 이를 자동으로 처리하는 방법 · 옵티마이저 SGD, 모멘텀, Adam이 그래디언트를 좋은 갱신으로 바꾸는 방법

## 문제: 수백만 개의 도함수

제8장에서는 매개변수가 두 개인 직선을 학습시켰고, 두 도함수를 손으로 적었습니다. 작은 언어 모델에도 매개변수가 수천만 개 있고, 큰 모델에는 수십억 개가 있습니다. 우리는 학습의 매 스텝마다 이 *모든* 매개변수에 대한 손실의 도함수가 필요합니다.

단순한 아이디어는 제6장의 수치 미분입니다. 매개변수 하나를 아주 조금 바꾸고, 신경망을 다시 실행해서, 손실이 어떻게 변하는지 보는 것입니다. 하지만 이렇게 하려면 *매개변수마다* 신경망 전체를 적어도 한 번 더 실행해야 합니다. 매개변수가 천만 개라면 한 스텝에 천만 번을 실행해야 합니다. 학습은 결코 끝나지 않을 것입니다.

**역전파**(backpropagation)는 이 문제를 해결합니다. 역전파는 신경망을 대략 한 번 더 통과하는 비용으로 모든 매개변수에 대한 정확한 그래디언트를 계산합니다. 이는 손실 자체를 계산하는 것과 비슷한 비용입니다(작은 상수배로, 흔히 순전파 한 번의 두세 배라고 합니다). 이 아이디어는 1980년대에 러멜하트(Rumelhart), 힌턴(Hinton), 윌리엄스(Williams)가 신경망에 널리 알렸고, 지금도 모든 현대 신경망이 이 방법으로 학습됩니다. 필요한 수학은 제6장의 연쇄 법칙뿐입니다.

## 계산 그래프와 연쇄 법칙

신경망의 모든 계산은 **계산 그래프**(computational graph)로 그릴 수 있습니다. 각 노드는 단순한 연산(곱셈, 덧셈, ReLU, 제곱)이고, 화살표는 어떤 결과가 어디로 들어가는지 보여 줍니다. 화살표를 따라 입력에서 출력을 계산하는 과정을 **순전파**(forward pass)라고 합니다.

**연쇄 법칙**(chain rule)을 다시 떠올려 봅시다. $y$가 $u$에 의존하고 $u$가 $x$에 의존하면 다음이 성립합니다.

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}.$$

말로 하면, $x$에 대한 $y$의 민감도는 중간값 $u$에 대한 $y$의 민감도에 $x$에 대한 $u$의 민감도를 곱한 것입니다. 긴 사슬은 그저 더 긴 곱일 뿐입니다. 어떤 변수가 여러 경로를 통해 출력에 영향을 준다면, 모든 경로의 기여를 더합니다.

역전파는 이 규칙을 그래프의 끝에서 처음으로 체계적으로 적용합니다. 손실에서 $\frac{\partial L}{\partial L} = 1$로 시작합니다. 그다음 노드를 하나씩 거슬러 올라가면서, 각 노드는 "내 출력에 대한 손실의 도함수"를 받아 자신의 **국소 도함수**(local derivative), 즉 "내 출력은 내 입력에 어떻게 의존하는가?"를 곱하고, 그 결과를 자신의 입력 쪽으로 넘깁니다. 이것이 **역방향 패스**(backward pass)입니다. 각 노드는 자신의 단순한 연산만 알면 되고, 연쇄 법칙이 이들을 이어 줍니다.

## 손으로 풀어 보는 예제

입력 하나, ReLU 은닉 뉴런 하나, 출력 하나로 이루어지고 편향이 없는 아주 작은 신경망을 생각해 봅시다.

$$u = w_1 x, \qquad h = \operatorname{ReLU}(u), \qquad \hat{y} = w_2 h, \qquad L = (\hat{y} - y)^2.$$

$x = 2$, 목표값 $y = 5$, 현재 가중치는 $w_1 = 1.5$, $w_2 = 2$라고 합시다.

**순전파.** $u = 1.5 \cdot 2 = 3$; $h = \operatorname{ReLU}(3) = 3$; $\hat{y} = 2 \cdot 3 = 6$; $L = (6 - 5)^2 = 1$.

**역방향 패스.** 순전파의 값을 그대로 둔 채, 손실에서 가중치 쪽으로 거슬러 갑니다.

1. $\dfrac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y) = 2$.
2. $\hat{y} = w_2 h$이므로 $\dfrac{\partial L}{\partial w_2} = \dfrac{\partial L}{\partial \hat{y}} \cdot h = 2 \cdot 3 = 6$이고, $\dfrac{\partial L}{\partial h} = \dfrac{\partial L}{\partial \hat{y}} \cdot w_2 = 2 \cdot 2 = 4$입니다.
3. ReLU의 국소 도함수는 입력이 양수이면 1, 음수이면 0입니다. 여기서는 $u = 3 > 0$이므로 $\dfrac{\partial L}{\partial u} = 4 \cdot 1 = 4$입니다.
4. $u = w_1 x$이므로 $\dfrac{\partial L}{\partial w_1} = \dfrac{\partial L}{\partial u} \cdot x = 4 \cdot 2 = 8$입니다.

따라서 그래디언트는 $(\partial L / \partial w_1,\ \partial L / \partial w_2) = (8,\ 6)$입니다. 둘 다 양수이므로 경사 하강법은 두 가중치를 모두 *줄일* 것이고, 그러면 $\hat{y}$가 6에서 목표값 5 쪽으로 내려갑니다. 두 가지를 눈여겨보십시오. 첫째, 역방향 패스는 순전파의 값($h$, $x$, $w_2$)을 다시 사용했습니다. 그래서 프레임워크는 중간 결과를 저장하며, 이 때문에 학습은 모델을 단순히 실행할 때보다 훨씬 많은 메모리를 필요로 합니다. 둘째, 중간값 $\partial L / \partial h$는 한 번만 계산된 뒤 그 앞의 모든 계산에 쓰였습니다. 바로 이런 재사용이 역전파를 효율적으로 만듭니다.

PyTorch의 **자동 미분**(automatic differentiation, autograd)은 바로 이 단계들을 대신 수행합니다.

```python
import torch

x, y = torch.tensor(2.0), torch.tensor(5.0)
w1 = torch.tensor(1.5, requires_grad=True)   # w1의 그래디언트를 추적
w2 = torch.tensor(2.0, requires_grad=True)   # w2도 추적

h = torch.relu(w1 * x)        # 순전파: h = 3
y_hat = w2 * h                # y_hat = 6
loss = (y_hat - y) ** 2       # loss = 1
loss.backward()               # 역방향 패스가 .grad를 채운다
print(w1.grad, w2.grad)       # tensor(8.) tensor(6.)
```

순전파 동안 PyTorch는 그래프를 기록하고, `loss.backward()`는 그 그래프를 거꾸로 따라가며 각 그래디언트를 매개변수의 `.grad` 필드에 저장합니다.

## 층을 통과하는 역전파와 그 위험

층 전체 $\mathbf{z} = W\mathbf{h} + \mathbf{b}$에 대해서도 같은 규칙이 벡터로 작동합니다. 위에서 내려온 그래디언트를 $\boldsymbol{\delta} = \partial L / \partial \mathbf{z}$라고 하면 다음과 같습니다.

$$\frac{\partial L}{\partial W} = \boldsymbol{\delta}\, \mathbf{h}^\top, \qquad \frac{\partial L}{\partial \mathbf{b}} = \boldsymbol{\delta}, \qquad \frac{\partial L}{\partial \mathbf{h}} = W^\top \boldsymbol{\delta}.$$

첫 번째 식은 **외적**(outer product)입니다. $(i, j)$ 원소는 $\delta_i h_j$이므로, 가중치는 자신의 입력과 자신의 출력이 받은 오차 신호에 비례해서 변합니다. 마지막 식은 전치 행렬(제6장)을 통해 오차를 아래 층으로 보냅니다. 따라서 역방향 패스는 순전파와 같은 구조를 가지며, 순서만 반대입니다.

깊은 신경망에서 처음 층들에 도달하는 그래디언트는 층마다 하나 이상씩 생기는 많은 국소 도함수의 곱입니다. 많은 수의 곱은 불안정합니다. 곱하는 값들이 대부분 1보다 작으면 그래디언트는 깊이에 따라 지수적으로 줄어듭니다. 이것이 **기울기 소실**(vanishing gradient) 문제입니다. 시그모이드의 도함수는 최대 0.25이므로, 시그모이드 층 열 개는 신호에 약 $0.25^{10} \approx 10^{-6}$을 곱할 수 있습니다. 곱하는 값들이 대부분 1보다 크면 그래디언트는 지수적으로 커집니다. 이것이 **기울기 폭발**(exploding gradient) 문제입니다. 이 문제들 때문에 ReLU(제7장), 신중한 **가중치 초기화**(weight initialization), 정규화(normalization) 층과 잔차 연결(제18장), 그래디언트 클리핑(제10장)이 등장했습니다. 이는 순환 신경망(제11장과 제12장)의 핵심적인 어려움이기도 합니다.

## 옵티마이저: 그래디언트에서 갱신으로

역전파는 그래디언트 $\mathbf{g} = \nabla_\theta L$을 줍니다. **옵티마이저**(optimizer)는 이를 이용해 매개변수를 어떻게 갱신할지 결정합니다. 꼭 알아야 할 옵티마이저는 세 가지입니다.

**SGD.** 제8장의 단순한 규칙 $\theta \leftarrow \theta - \eta\, \mathbf{g}$입니다. 실제로는 데이터의 작은 무작위 부분집합인 미니배치(제10장)에서 그래디언트를 계산하므로 그래디언트에 잡음이 섞입니다. 이것이 **확률적 경사 하강법**(stochastic gradient descent, SGD)입니다. 약점은 길고 좁은 골짜기에서 드러납니다. 골짜기를 가로지르는 방향은 경사가 가파르고, 골짜기를 따라가는 방향은 완만합니다. 골짜기를 따라 빨리 움직일 만큼 학습률을 키우면 매개변수가 골짜기의 이쪽 벽과 저쪽 벽 사이를 오가며 튀고, 안전하게 작은 학습률을 쓰면 기어가듯 느립니다.

**모멘텀.** 골짜기를 굴러 내려가는 무거운 공을 상상해 보십시오. 공은 속도를 유지하므로, 옆으로 가해지는 무작위 힘은 서로 상쇄되고 일정한 방향의 힘은 쌓입니다. 우리는 누적 "속도" $\mathbf{v}$를 유지합니다.

$$\mathbf{v} \leftarrow \beta\, \mathbf{v} + \mathbf{g}, \qquad \theta \leftarrow \theta - \eta\, \mathbf{v}.$$

여기서 $\beta$(보통 0.9)는 이전 속도를 얼마나 유지할지 정합니다. 이 방법을 **모멘텀**(momentum)이라고 합니다. 좁은 골짜기에서 골짜기를 가로지르는 그래디언트는 매 스텝 부호가 바뀌어 상쇄되고, 골짜기를 따라가는 그래디언트는 같은 방향을 가리키므로 누적됩니다.

**Adam**(킹마(Kingma)와 바(Ba))은 모멘텀에 매개변수마다 따로 적응하는 보폭을 결합합니다. Adam은 그래디언트와 그래디언트 제곱의 누적 평균 두 가지를 유지합니다.

$$\mathbf{m} \leftarrow \beta_1 \mathbf{m} + (1 - \beta_1)\, \mathbf{g}, \qquad \mathbf{v} \leftarrow \beta_2 \mathbf{v} + (1 - \beta_2)\, \mathbf{g}^2,$$

$$\hat{\mathbf{m}} = \frac{\mathbf{m}}{1 - \beta_1^t}, \qquad \hat{\mathbf{v}} = \frac{\mathbf{v}}{1 - \beta_2^t}, \qquad \theta \leftarrow \theta - \eta\, \frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{v}}} + \epsilon}.$$

제곱 $\mathbf{g}^2$, 제곱근, 나눗셈은 성분별로 계산합니다. $\mathbf{m}$은 모멘텀("1차 모멘트")이고, $\mathbf{v}$는 그래디언트 각 성분의 전형적인 크기("2차 모멘트")를 추정합니다. $\sqrt{\hat{\mathbf{v}}}$로 나누면, 그래디언트가 크든 작든 각 매개변수는 대략 $\eta$ 크기의 스텝을 밟게 됩니다. 가파른 방향은 느려지고 완만한 방향은 빨라집니다. 문자 위의 모자 기호는 **편향 보정**(bias correction)입니다. $\mathbf{m}$과 $\mathbf{v}$는 0에서 시작하므로, $1 - \beta^t$($t$는 스텝 번호)로 나누어 초기에 0 쪽으로 치우친 편향을 없앱니다. $\epsilon$은 $10^{-8}$ 같은 아주 작은 수로, 0으로 나누는 것을 막습니다. 일반적인 기본값은 $\beta_1 = 0.9$, $\beta_2 = 0.999$입니다. Adam, 특히 가중치 감쇠를 분리한 변형인 **AdamW**(제10장)는 트랜스포머와 언어 모델 학습의 표준 옵티마이저입니다. 대가는 메모리입니다. 매개변수마다 숫자 두 개를 추가로 저장해야 합니다.

세 옵티마이저를 좁은 골짜기 $f(x, y) = x^2 + 20y^2$에서 비교해 봅시다. 시작점은 $f = 36$인 $(-4, 1)$입니다.

```python
import torch

def f(p):                                  # 길고 좁은 골짜기:
    return p[0] ** 2 + 20 * p[1] ** 2      # x 방향은 완만, y 방향은 가파름

def run(make_opt, steps=100):
    p = torch.tensor([-4.0, 1.0], requires_grad=True)
    opt = make_opt([p])
    for _ in range(steps):
        opt.zero_grad()                    # 이전 그래디언트를 지운다
        loss = f(p)
        loss.backward()                    # 새 그래디언트를 계산한다
        opt.step()                         # 매개변수를 갱신한다
    return round(f(p).item(), 4)

print("SGD lr=0.01 ", run(lambda ps: torch.optim.SGD(ps, lr=0.01)))
print("SGD lr=0.05 ", run(lambda ps: torch.optim.SGD(ps, lr=0.05)))
print("Momentum    ", run(lambda ps: torch.optim.SGD(ps, lr=0.01, momentum=0.9)))
print("Adam        ", run(lambda ps: torch.optim.Adam(ps, lr=0.1)))
```

100 스텝 후 결과는 다음과 같습니다. `lr=0.01`인 SGD는 약 0.28로, 아직 완만한 방향을 따라 기어가고 있습니다. `lr=0.05`인 SGD는 정확히 20.0입니다. $y$ 방향에서 매 스텝 $y$에 $1 - 0.05 \cdot 40 = -1$이 곱해지므로 $y = 1$과 $y = -1$ 사이를 영원히 오갑니다. 모멘텀은 약 0.0008, Adam은 약 0.0006입니다. `zero_grad()` → `backward()` → `step()`이라는 패턴은 모든 PyTorch 학습 루프의 핵심이며, 제10장과 그 이후의 모든 장에서 사용합니다. PyTorch는 기본적으로 새 그래디언트를 기존 그래디언트에 *더하기* 때문에 `zero_grad()`가 필요하다는 점에 유의하십시오.

## 요약
- 역전파는 손실에서부터 거꾸로 연쇄 법칙을 적용하여, 대략 한 번의 추가 패스 비용으로 모든 매개변수에 대한 그래디언트를 계산합니다.
- 순전파는 중간값을 계산하고 저장하며, 역방향 패스는 국소 도함수를 곱하고 공통 결과를 재사용합니다.
- PyTorch autograd는 계산 그래프를 기록하고, `loss.backward()`를 호출하면 `.grad`를 채웁니다.
- 국소 도함수의 긴 곱은 깊은 신경망에서 기울기 소실이나 기울기 폭발을 일으킬 수 있습니다.
- SGD는 그래디언트 반대 방향으로 이동하고, 모멘텀은 속도를 누적하며, Adam은 매개변수별 적응적 보폭을 더한 것으로 트랜스포머의 표준 선택입니다.

## 연습 문제
1. 역전파가 매개변수마다 수치 미분을 계산하는 것보다 훨씬 빠른 이유를 자신의 말로 설명해 보십시오.
2. $w_1 = -1$로 바꾸고(나머지 값은 그대로) 예제를 다시 풀어 보십시오. $h$는 얼마입니까? 두 그래디언트는 각각 얼마입니까? 이것은 "죽은" ReLU 뉴런에 대해 무엇을 알려 줍니까?
3. 출력에 편향 $b$를 더해 $\hat{y} = w_2 h + b$로 만들고 $b = 0.5$라고 합시다. $\partial L / \partial b$를 손으로 계산하고 PyTorch로 확인해 보십시오.
4. 옵티마이저 코드에서 `lr=0.045`인 SGD와 `momentum=0.5`인 모멘텀을 시도해 보십시오. 좁은 골짜기라는 아이디어로 결과를 설명해 보십시오.

## 핵심 용어
| 한국어 | English | Русский | 러시아어 발음 |
|---|---|---|---|
| 역전파 | backpropagation | метод обратного распространения ошибки | métod obrátnogo rasprostranéniya oshíbki |
| 계산 그래프 | computational graph | вычислительный граф | vychislítel'nyy graf |
| 연쇄 법칙 | chain rule | цепное правило | tsepnóye právilo |
| 순전파 | forward pass | прямой проход | pryamóy prokhód |
| 역방향 패스 | backward pass | обратный проход | obrátnyy prokhód |
| 국소 도함수 | local derivative | локальная производная | lokál'naya proizvódnaya |
| 자동 미분 | automatic differentiation | автоматическое дифференцирование | avtomatícheskoye differentsírovaniye |
| 기울기 소실 | vanishing gradient | затухающий градиент | zatukháyushchiy gradiyént |
| 기울기 폭발 | exploding gradient | взрывающийся градиент | vzryváyushchiysya gradiyént |
| 옵티마이저 | optimizer | оптимизатор | optimizátor |
| 확률적 경사 하강법 | stochastic gradient descent | стохастический градиентный спуск | stokhastícheskiy gradiyéntnyy spusk |
| 모멘텀 | momentum | импульс | ímpul's |
| 편향 보정 | bias correction | коррекция смещения | korréktsiya smeshchéniya |
| 가중치 초기화 | weight initialization | инициализация весов | initsializátsiya vesóv |

---
[← 이전: 손실 함수와 경사 하강법](08-loss-and-gradient-descent.md) · [목차](README.md) · [다음: 실전 학습: 배치, 과적합, 정규화 →](10-training-in-practice.md)
