# 제18장. 트랜스포머 블록: 잔차 연결, LayerNorm/RMSNorm, FFN, GLU

> **제4부 — 트랜스포머** · 수준: 중급
>
> **이 장에서 배울 내용:** 트랜스포머 블록 하나가 어떤 부품으로 이루어지며 각 부품이 왜 필요한가 · 잔차 연결과 정규화가 어떻게 깊은 스택을 학습 가능하게 만드는가 · 피드포워드 네트워크가 무엇을 하며 현대 모델이 왜 SwiGLU 같은 게이트 변형을 쓰는가

## 왜 어텐션만으로는 부족한가

제16장의 멀티헤드 어텐션 층에 제17장의 위치 정보를 더하고, 이런 층을 그냥 24개 쌓는다고 해 봅시다. 좋은 언어 모델이 될까요? 그렇지 않습니다. 이유는 두 가지입니다.

첫 번째 이유는 층들이 *무엇을* 계산하는가에 관한 것입니다. 어텐션은 값 벡터들의 가중 평균을 출력합니다. 평균은 토큰 *사이에서* 정보를 옮기는 데("주어의 무언가를 동사로 복사하기")는 좋지만, 각 토큰 내부의 정보를 *변환*하는 일은 거의 하지 않습니다. softmax를 제외하면 어텐션은 대체로 값에 대해 선형입니다. 제7장의 다층 신경망처럼 각 토큰의 벡터를 받아 진정한 비선형성으로 처리하는 부품이 따로 필요합니다.

두 번째 이유는 *학습*에 관한 것입니다. 제9장과 제11장에서 보았듯이 매우 깊은 신경망은 기울기 소실과 폭발, 그리고 층마다 활성값의 크기가 흔들리는 문제를 겪습니다. 특별한 대책 없이 24개 층을 쌓으면 거의 학습되지 않을 것입니다.

**트랜스포머 블록(Transformer block)**은 두 문제를 모두 해결합니다. 블록에는 두 개의 **하위 층(sublayer)**이 있습니다. 토큰들이 *소통*하게 하는 멀티헤드 어텐션과, 각 토큰이 *계산*하게 하는 **피드포워드 네트워크(feed-forward network, FFN)**입니다. 각 하위 층 주위에는 두 가지 공학적 장치, 즉 **잔차 연결(residual connection)**과 **정규화(normalization)** 층이 놓입니다. 모델은 같은 블록을 $N$번 반복한 것에 불과합니다. 이 장에서는 블록을 분해해 봅니다.

## 잔차 연결과 잔차 스트림

잔차 연결은 He 등이 이미지 신경망(ResNet)을 위해 도입한 간단한 아이디어입니다. 입력 $x$를 하위 층의 출력 $F(x)$로 *교체하는* 대신, 출력을 입력에 *더합니다*.

$$
x_{\text{new}} = x + F(x)
$$

이것이 왜 도움이 될까요? 기울기를 살펴봅시다. 연쇄 법칙(제9장)에 따라 $\partial x_{\text{new}} / \partial x = I + \partial F / \partial x$이며, 여기서 $I$는 단위 행렬입니다. 하위 층 자체의 도함수가 아주 작더라도 단위 행렬 항이 기울기를 그대로 뒤로 전달합니다. 모든 블록에 잔차 연결이 있으면 손실에서 첫 번째 층까지 곧바로 이어지는 "고속도로"가 생기므로, 신경망이 깊다는 이유만으로 기울기가 소실되지는 않습니다. 두 번째 장점도 있습니다. 학습 초기에 $F$가 0에 가까우면 각 블록은 항등 함수에 가깝습니다. 따라서 깊은 신경망은 거의 자명한 신경망에서 출발하여 점차 자신의 층을 활용하는 법을 배웁니다.

잔차 연결은 모델 전체를 바라보는 유용한 관점도 줍니다. 토큰마다 벡터 하나가 임베딩 층에서 출력까지 흘러간다고 상상해 보십시오. 모든 어텐션 하위 층과 FFN 하위 층은 이 벡터를 *읽고*, 덧셈으로 업데이트를 다시 *씁니다*. 이 누적 합을 **잔차 스트림(residual stream)**이라고 합니다. 이 관점에서 층들은 다음 층에 메시지를 전달하기만 하는 것이 아니라, 이후의 모든 층이 읽을 수 있는 공유 메모리에 기록합니다. 이 관점은 해석 가능성 연구(제40장)에서 매우 흔히 쓰입니다.

## LayerNorm과 RMSNorm

층마다 업데이트를 더하다 보면 잔차 스트림의 벡터가 커지거나 크기가 달라질 수 있습니다. 신경망은 각 층의 입력 크기가 안정적일 때 가장 잘 학습되므로, 트랜스포머는 각 하위 층 앞(또는 뒤)에서 벡터를 정규화합니다.

Ba, Kiros, Hinton이 제안한 **층 정규화(layer normalization, LayerNorm)**는 각 토큰의 벡터를 그 $d$개 특성에 대해 따로 정규화합니다.

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta,
\qquad \mu = \frac1d \sum_{m=1}^{d} x_m,\quad \sigma^2 = \frac1d \sum_{m=1}^{d} (x_m - \mu)^2
$$

여기서 $\mu$와 $\sigma^2$는 $x$ 성분들의 평균과 분산이고, $\epsilon$은 0으로 나누는 것을 막는 아주 작은 수이며, $\gamma$와 $\beta$는 크기 $d$인 학습 가능한 벡터(스케일과 이동)이고, $\odot$는 **원소별 곱(element-wise multiplication)**을 뜻합니다. 배치 정규화(제10장)와 달리 LayerNorm은 배치의 다른 예제에 의존하지 않으므로 길이가 다른 시퀀스나 토큰을 하나씩 생성하는 경우에 잘 맞습니다.

Zhang과 Sennrich의 **RMSNorm**은 평균 빼기와 이동 $\beta$를 없애 LayerNorm을 단순화합니다. 성분들의 **제곱평균제곱근(root mean square)**으로 나누기만 합니다.

$$
\text{RMSNorm}(x) = g \odot \frac{x}{\sqrt{\frac1d \sum_{m=1}^{d} x_m^2 + \epsilon}}
$$

여기서 $g$는 학습 가능한 이득 벡터입니다. RMSNorm은 조금 더 저렴하면서도 실제로는 똑같이 잘 동작하므로 LLaMA를 비롯한 많은 최근 모델에서 사용됩니다.

$x = (1, 2, 3, 4)$로 예를 들어 봅시다. $\epsilon$은 무시하고 $\gamma = g = 1$, $\beta = 0$으로 둡니다. LayerNorm: $\mu = 2.5$, $\sigma^2 = (2.25 + 0.25 + 0.25 + 2.25)/4 = 1.25$, $\sigma \approx 1.118$이므로 출력은 $(-1.342, -0.447, 0.447, 1.342)$이며, 평균 0, 분산 1입니다. RMSNorm: 제곱의 평균은 $(1 + 4 + 9 + 16)/4 = 7.5$, 제곱평균제곱근은 $\approx 2.739$이므로 출력은 $(0.365, 0.730, 1.095, 1.461)$입니다. 방향은 $x$와 같지만 크기가 표준화되었습니다.

**정규화를 어디에 둘까요?** 원래의 트랜스포머는 **사후 정규화(Post-Norm)**, 즉 $x_{\text{new}} = \text{LayerNorm}(x + F(x))$를 사용하여 잔차 덧셈 *이후에* 정규화했습니다. 이후 연구에서 **사전 정규화(Pre-Norm)**, 즉 $x_{\text{new}} = x + F(\text{Norm}(x))$가 특히 깊은 모델에서 더 안정적으로 학습되고 학습률 워밍업에 덜 민감하다는 것이 밝혀졌습니다. 이유는 식에서 보입니다. Pre-Norm에서는 입력에서 출력으로 가는 잔차 경로에 정규화가 전혀 없으므로 기울기 "고속도로"가 완전히 깨끗합니다. 거의 모든 현대 LLM은 Pre-Norm을 쓰고, 마지막 블록 뒤에 정규화를 하나 더 둡니다.

## 피드포워드 네트워크와 게이트 변형

FFN은 각 토큰의 벡터에 *독립적으로*, 모든 위치에 같은 가중치로 적용되는 작은 2층 신경망입니다.

$$
\text{FFN}(x) = W_2\, \sigma(W_1 x + b_1) + b_2
$$

여기서 크기 $d_{\text{ff}} \times d$인 $W_1$은 벡터를 보통 $4d$인 더 큰 **은닉 차원(hidden dimension)** $d_{\text{ff}}$로 확장합니다. $\sigma$는 **활성화 함수(activation function)**(원 논문에서는 ReLU, BERT와 GPT에서는 GELU)이고, 크기 $d \times d_{\text{ff}}$인 $W_2$는 다시 폭 $d$로 투영합니다. 소박해 보이지만 FFN은 파라미터의 대부분을 담고 있습니다. $d_{\text{ff}} = 4d$일 때 두 행렬에는 $8d^2$개의 가중치가 있고, 어텐션의 네 행렬에는 $4d^2$개가 있습니다. 해석 가능성 연구가 뒷받침하는 유용한 직관은 FFN 층이 거대한 키–값 메모리처럼 동작한다는 것입니다. $W_1$의 행들은 토큰 벡터에서 패턴을 감지하고, 그에 대응하는 $W_2$의 열들은 연관된 정보를 잔차 스트림에 씁니다.

**게이트 선형 유닛(gated linear unit, GLU)**은 곱셈 게이트로 FFN을 개선합니다. Shazeer("GLU Variants Improve Transformer")는 여러 변형을 비교했으며, 그중 **SwiGLU**라는 변형이 LLaMA, PaLM 등 많은 모델의 표준이 되었습니다.

$$
\text{SwiGLU}(x) = W_{\text{down}} \big(\, \text{Swish}(W_{\text{gate}}\, x) \odot W_{\text{up}}\, x \,\big)
$$

이제 행렬이 세 개입니다. $W_{\text{up}} x$는 후보 특성을 계산합니다. $\text{Swish}(z) = z \cdot \text{sigmoid}(z)$(SiLU라고도 함)를 쓰는 $\text{Swish}(W_{\text{gate}} x)$는 각 특성을 "얼마나 통과시킬지"를 부드럽게 계산합니다. 둘의 원소별 곱이 $W_{\text{down}}$을 통과합니다. 게이트 덕분에 신경망은 한 특성이 다른 특성에 곱셈적으로 의존하게 만들 수 있는데, 일반 ReLU나 GELU로는 이것이 쉽지 않습니다. 파라미터 수를 표준 FFN과 같게 유지하려고 은닉 차원을 약 $\tfrac{8}{3} d$로 줄입니다. 그러면 $3 \cdot d \cdot \tfrac83 d = 8d^2$이 됩니다.

## 블록 조립하기

다음은 현대 디코더 모델 스타일로 RMSNorm과 SwiGLU를 쓴 완전한 Pre-Norm 블록입니다. 간결함을 위해 제16장의 클래스 대신 PyTorch 내장 `nn.MultiheadAttention`을 사용합니다. 함정이 하나 있으니 주의하십시오. PyTorch의 규약에서 불리언 `attn_mask`는 *차단된* 위치를 `True`로 표시하며, 이는 제16장에서 쓴 규약과 반대입니다.

```python
import torch
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.g = eps, nn.Parameter(torch.ones(d))  # 학습 가능한 이득 g
    def forward(self, x):
        return self.g * x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.w_down(nn.functional.silu(self.w_gate(x)) * self.w_up(x))  # 게이트 * 값
class Block(nn.Module):                                   # Pre-Norm 트랜스포머 블록
    def __init__(self, d, n_heads, hidden):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ffn = SwiGLU(d, hidden)
    def forward(self, x, mask=None):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]  # 잔차 연결 1
        return x + self.ffn(self.norm2(x))                                  # 잔차 연결 2

mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), 1)  # PyTorch: True = 차단
block = Block(48, 4, 128)                                 # d = 48, 헤드 4개, hidden = 8/3 * 48 = 128
print(block(torch.randn(2, 5, 48), mask).shape, sum(p.numel() for p in block.parameters()))
```

출력은 `torch.Size([2, 5, 48]) 27936`입니다. 모양은 변하지 않습니다. 블록은 벡터 시퀀스를 같은 크기의 벡터 시퀀스로 바꾸며, 바로 그 때문에 블록을 쌓을 수 있습니다. 파라미터는 다음과 같이 나뉩니다. 어텐션 $4 \cdot 48^2 + 4 \cdot 48 = 9408$(행렬 네 개와 편향), SwiGLU $3 \cdot 48 \cdot 128 = 18432$, RMSNorm 이득 두 개 $2 \cdot 48 = 96$입니다. 예상대로 FFN이 블록의 약 3분의 2를 차지합니다.

블록 하나의 데이터 흐름을 두 줄로 정리하면 다음과 같습니다.

$$
x \leftarrow x + \text{MHA}(\text{Norm}(x)), \qquad x \leftarrow x + \text{FFN}(\text{Norm}(x)).
$$

완전한 모델은 다음과 같습니다. 토큰 임베딩(여기에 위치를 더하거나 어텐션 안에서 RoPE 사용) → 블록 $N$개 → 최종 정규화 → 각 벡터를 어휘에 대한 로짓으로 바꾸는 선형층. 제19장에서는 블록이 인코더와 디코더로 어떻게 배치되는지 보고, 제20장에서는 이런 모델을 직접 만들고 학습합니다.

## 요약

- 트랜스포머 블록에는 두 하위 층이 있습니다. 멀티헤드 어텐션(토큰 사이의 소통)과 피드포워드 네트워크(각 토큰 내부의 계산)입니다.
- 잔차 연결은 각 하위 층의 출력을 입력에 더합니다. 깊은 스택에서 기울기가 계속 흐르게 하고 잔차 스트림을 형성합니다.
- LayerNorm은 각 토큰의 벡터를 평균 0, 분산 1로 정규화합니다. RMSNorm은 제곱평균제곱근으로 나누기만 하며 더 저렴합니다.
- Pre-Norm($x + F(\text{Norm}(x))$)은 원래의 Post-Norm보다 안정적으로 학습되며 거의 모든 현대 LLM에서 사용됩니다.
- FFN은 약 $4d$의 은닉 차원으로 확장하며 파라미터의 대부분을 담습니다. SwiGLU는 곱셈 게이트를 더하고 약 $\tfrac83 d$의 은닉 크기를 씁니다.

## 연습 문제

1. 블록 안에서 어텐션이 하는 일과 FFN이 하는 일의 차이를 자신의 말로 설명해 보십시오.
2. $x = (2, 2, 2, 6)$에 LayerNorm과 RMSNorm을 손으로 적용해 보십시오($\gamma = g = 1$, $\beta = 0$, $\epsilon = 0$).
3. $d = 4096$일 때 표준 FFN($d_{\text{ff}} = 4d$)을 쓰는 블록 하나의 파라미터 수는 얼마입니까(편향과 정규화 제외)? SwiGLU($d_{\text{ff}} = \tfrac83 d$)라면 어떻습니까?
4. `Block` 클래스를 Post-Norm으로 바꾸고, 두 버전을 블록 24개짜리 스택으로 무작위 데이터에서 수백 스텝 학습해 보십시오. 높은 학습률에서 손실이 어떻게 변하는지 비교해 보십시오.

## 핵심 용어

| 한국어 | English | Русский | 러시아어 발음 |
|---|---|---|---|
| 트랜스포머 블록 | Transformer block | блок трансформера | blok transfórmera |
| 하위 층 | sublayer | подслой | podslóy |
| 잔차 연결 | residual connection | остаточная связь | ostátochnaya svyaz' |
| 잔차 스트림 | residual stream | остаточный поток | ostátochnyy potók |
| 층 정규화 | layer normalization | нормализация слоя | normalizátsiya slóya |
| 제곱평균제곱근 | root mean square | среднеквадратичное значение | srednekvadratíchnoye znachéniye |
| 원소별 곱 | element-wise multiplication | поэлементное умножение | poeleméntnoye umnozhéniye |
| 사전 정규화 | Pre-Norm | преднормализация | prednormalizátsiya |
| 피드포워드 네트워크 | feed-forward network | полносвязная сеть прямого распространения | polnosvyáznaya set' pryamógo rasprostranéniya |
| 은닉 차원 | hidden dimension | скрытая размерность | skrýtaya razmérnost' |
| 활성화 함수 | activation function | функция активации | fúnktsiya aktivátsii |
| 게이트 선형 유닛 | gated linear unit (GLU) | вентильный линейный блок | véntil'nyy linéynyy blok |
| 단위 행렬 | identity matrix | единичная матрица | yediníchnaya mátritsa |
| 이득 벡터 | gain vector | вектор усиления | véktor usiléniya |

---
[← 이전: 위치 정보: 사인파 인코딩, 학습형 인코딩, RoPE, ALiBi](17-positional-encoding.md) · [목차](README.md) · [다음: 인코더, 디코더, 인코더–디코더 아키텍처 →](19-transformer-architectures.md)
