# 제32장. 매개변수 효율적 미세 조정: LoRA와 QLoRA

> **제7부 — 모델의 적응과 정렬** · 수준: 고급
>
> **이 장에서 배울 내용:** 전체 미세 조정이 비싼 이유와 그것이 종종 불필요한 이유 · LoRA가 고정된 가중치에 학습 가능한 저랭크 업데이트를 더하는 방식 · QLoRA가 LoRA와 4비트 양자화를 결합하여 GPU 한 대로 큰 모델을 미세 조정하는 방법

## 정말 모든 가중치를 바꿔야 할까?

31장에서 우리는 모델의 모든 파라미터를 미세 조정했습니다. 이것이 70억 개 파라미터 모델에 무엇을 의미하는지 생각해 봅시다. 혼합 정밀도(28장)에서 Adam을 쓰면 파라미터 하나당 대략 16바이트가 필요합니다. 16비트 형식의 가중치와 그래디언트, 그리고 32비트 마스터 사본과 32비트 Adam 상태 두 개입니다. 활성값을 세기도 전에 약 $7 \cdot 10^9 \cdot 16 \approx 112$GB로, 일반적인 GPU 한 대가 가진 메모리보다 많습니다. 게다가 열 가지 과제에 대해 미세 조정하면 모델의 전체 사본을 열 개 저장해야 합니다.

그런데 우리가 원하는 변화는 대개 작습니다. 모델에게 러시아어를 처음부터 가르치는 것이 아니라, 형식이나 문체, 새로운 레이블 집합을 가르치는 것입니다. 미세 조정의 **내재 차원**(intrinsic dimension)에 관한 실험(Aghajanyan 외)은 고정된 무작위 행렬로 전체 가중치 공간에 투영되는 놀라울 만큼 적은 수의 파라미터만 최적화해도 많은 과제를 학습할 수 있음을 발견했습니다. 이로부터 하나의 가설이 나옵니다. 미세 조정이 가중치 행렬에 가하는 *업데이트* $\Delta W$는 낮은 "유효 랭크"를 가진다는 것입니다. **매개변수 효율적 미세 조정**(parameter-efficient fine-tuning, PEFT) 방법은 이 점을 이용합니다. 사전 학습된 모델을 고정하고 소수의 추가 파라미터만 학습시키는 것입니다.

## 1분 만에 보는 저랭크 행렬

6장에서 배웠듯이 행렬의 **랭크**(rank, 계수)는 선형 독립인 행(또는 열)의 개수입니다. 랭크가 $r$인 행렬 $M \in \mathbb{R}^{d \times k}$는 언제나 다음과 같은 곱으로 쓸 수 있습니다.

$$M = B A, \qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k}.$$

$r$이 $d$와 $k$보다 훨씬 작다면, 인수들을 저장하는 것이 $M$ 자체를 저장하는 것보다 훨씬 저렴합니다. $dk$개 대신 $r(d + k)$개의 수만 있으면 됩니다. 예를 들어 $d = k = 4096$, $r = 8$이면 다음과 같습니다.

- 전체 행렬: $4096 \cdot 4096 = 16{,}777{,}216$개의 수
- 저랭크 인수: $8 \cdot (4096 + 4096) = 65{,}536$개의 수, 즉 약 $0.39\%$

작은 수치 예제로 개념을 구체화해 봅시다. $B = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$, $A = \begin{pmatrix} 1 & 0 & -1 \end{pmatrix}$라고 합시다. 둘의 곱은 모든 행이 $(1, 0, -1)$의 배수인 $3 \times 3$ 행렬입니다.

$$BA = \begin{pmatrix} 1 & 0 & -1 \\ 2 & 0 & -2 \\ 3 & 0 & -3 \end{pmatrix}.$$

수는 아홉 개이지만 이를 기술하는 데는 여섯 개만 필요했고, 이 행렬의 랭크는 1입니다. **저랭크 행렬**(low-rank matrix)은 "단순한" 변화만 표현할 수 있지만, LoRA의 가설은 미세 조정에는 단순한 변화로 충분하다는 것입니다.

## LoRA: 저랭크 적응

**LoRA**(low-rank adaptation, Hu 외, 2021)는 이 아이디어를 트랜스포머에서 선택한 각 가중치 행렬에 적용합니다. $W_0 \in \mathbb{R}^{d \times k}$를 고정된 사전 학습 가중치(예: 15장의 쿼리 투영 $W_Q$)라고 합시다. LoRA는 전체 업데이트를 학습하는 대신 두 개의 작은 행렬 $B \in \mathbb{R}^{d \times r}$와 $A \in \mathbb{R}^{r \times k}$를 학습하고 다음을 계산합니다.

$$h = W_0 x + \frac{\alpha}{r} B A x,$$

여기서 $x \in \mathbb{R}^k$는 입력, $h \in \mathbb{R}^d$는 출력, $r$은 어댑터의 랭크(일반적인 값은 4에서 64), $\alpha$는 상수 **스케일링 계수**(scaling factor)입니다. $r$로 나누기 때문에 $r$을 바꿀 때 학습률을 그만큼 많이 다시 조정할 필요가 없습니다.

세 가지 세부 사항이 중요합니다.

1. **초기화.** $A$는 작은 무작위 값으로, $B$는 0으로 초기화합니다. 따라서 처음에는 $BA = 0$이고, 모델은 사전 학습된 모델과 *정확히* 똑같이 동작합니다. 일반적인 미세 조정처럼 학습은 사전 학습된 해에서 시작합니다.
2. **$A$와 $B$만 학습합니다.** $W_0$는 그래디언트를 받지 않으므로 옵티마이저 상태도 필요 없습니다. 그래디언트와 Adam 상태를 위한 메모리가 몇 자릿수나 줄어듭니다.
3. **병합.** 학습이 끝나면 $W = W_0 + \frac{\alpha}{r} BA$를 한 번 계산해 $W_0$를 대체할 수 있습니다. 병합된 모델은 원래 모델과 모양도 속도도 같습니다. LoRA는 **추론 지연을 추가하지 않습니다**. 또는 기본 모델 하나와 과제별 작은 어댑터 여러 개를 두고 그 사이를 전환할 수도 있습니다.

원 논문에서 LoRA는 어텐션 투영(특히 $W_Q$와 $W_V$)에 적용되었습니다. 실제로는 피드포워드 네트워크(18장)를 포함한 모든 선형 층에 적용하는 편이 더 잘 작동하는 경우가 많습니다.

다음은 PyTorch로 작성한 완전한 LoRA 층입니다.

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():          # W0와 편향을 고정
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # 처음에는 B = 0
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)

    def merge(self):                              # W = W0 + (alpha/r) * B A
        return self.base.weight + self.scale * self.B @ self.A

torch.manual_seed(0)
base = nn.Linear(512, 512)
lora = LoRALinear(base, r=8)
x = torch.randn(4, 512)
print(torch.allclose(lora(x), base(x)))           # True: 초기화 시점에는 출력이 동일
trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
print(trainable, base.weight.numel())              # 8192 262144  (약 3%)
with torch.no_grad():
    lora.B.normal_()                               # B가 학습되었다고 가정
    merged = lora.merge()
print(torch.allclose(lora(x), x @ merged.T + base.bias, atol=1e-4))  # True
```

마지막 줄은 병합 성질을 확인합니다. 어댑터 경로와 병합된 단일 행렬이 같은 출력을 냅니다. $512 \times 512$ 층에서 어댑터의 파라미터는 $8 \cdot (512 + 512) = 8192$개로, $262{,}144$개 가중치의 약 $3\%$입니다. 위의 $4096 \times 4096$ 예처럼 층이 클수록 이 비율은 더 작아집니다.

## QLoRA: 양자화된 모델의 미세 조정

LoRA는 그래디언트와 옵티마이저 상태를 위한 메모리 대부분을 없애 주지만, 고정된 기본 가중치는 여전히 저장해야 합니다. 16비트 형식에서 70억 개 파라미터 모델은 약 14GB를 차지하고, 더 큰 모델은 훨씬 더 많이 필요합니다. **QLoRA**(Dettmers 외, 2023)는 고정된 기본 모델을 4비트로 저장하고 그 위에서 LoRA 어댑터를 학습시켜 이 문제를 해결합니다. 어댑터와 연산은 16비트 정밀도(예: bfloat16)로 유지되며, 4비트 가중치는 행렬 곱셈에 필요할 때 블록 단위로 즉석에서 **역양자화**(dequantization)됩니다. 그래디언트는 양자화된 가중치를 *통과하여* 어댑터로 흐르지만, 양자화된 가중치 자체는 결코 갱신되지 않습니다.

**양자화**(quantization)란 수를 더 적은 비트로 표현하는 것입니다. 추론과 관련해서는 36장에서 자세히 공부합니다. 가장 간단한 방식은 블록 단위의 *absmax* 양자화입니다. 가중치를 예컨대 64개씩 블록으로 나누고, 각 블록에서 가장 큰 절댓값 $m$을 찾은 뒤, 각 가중치 $w$를 $[-7, 7]$ 범위의 정수 $q = \mathrm{round}(7 \cdot w / m)$로 대응시킵니다. 역양자화할 때는 $\hat w = q \cdot m / 7$을 계산합니다. 작은 정수들과 블록당 하나의 스케일만 저장하면 됩니다.

```python
import torch

torch.manual_seed(0)
w = torch.randn(4096)                           # 한 층의 가중치 (1차원으로 펼침)
blocks = w.view(-1, 64)                         # 64개씩 묶은 블록
m = blocks.abs().max(dim=1, keepdim=True).values  # 블록당 스케일 하나
q = torch.round(7 * blocks / m).to(torch.int8)  # 4비트 범위: 정수 -7..7
w_hat = (q.float() * m / 7).view(-1)            # 역양자화

print(q.min().item(), q.max().item())           # -7 7
print(f"mean abs error: {(w - w_hat).abs().mean():.3f}")  # 약 0.094
```

(간단히 하기 위해 정수를 `int8`로 저장했습니다. 실제 구현은 4비트 값 두 개를 한 바이트에 묶어 넣습니다.) QLoRA는 이 아이디어를 세 가지 방향으로 개선했습니다.

- **4비트 NormalFloat(NF4).** 사전 학습된 가중치는 대략 정규 분포를 따르므로, NF4는 등간격의 16개 수준 대신 정규 분포의 분위수에 수준을 배치합니다. 그러면 각 수준이 거의 같은 빈도로 사용되어 오차가 줄어듭니다.
- **이중 양자화**(double quantization). 블록별 스케일 $m$ 자체도 양자화하여 파라미터당 비트를 조금 더 절약합니다.
- **페이지드 옵티마이저**(paged optimizer). 예를 들어 매우 긴 시퀀스에서 생기는 일시적인 메모리 급증을 견디기 위해 옵티마이저 상태를 GPU와 CPU 메모리 사이에서 자동으로 옮길 수 있습니다.

이 기법들로 QLoRA 저자들은 650억 개 파라미터 모델을 48GB GPU 한 대에서 미세 조정했으며, 그들의 실험에서 품질은 16비트 전체 미세 조정에 가까웠습니다. 하드웨어가 제한된 박사 과정 학생에게 이는 큰 모델을 적응시키는 가장 실용적인 방법인 경우가 많습니다.

## 다른 PEFT 방법과 실용적인 조언

LoRA는 가장 인기 있는 PEFT 방법이지만 유일한 방법은 아닙니다.

- **어댑터**(adapter, Houlsby 외, 2019)는 각 트랜스포머 블록 안에 작은 병목 네트워크(축소 투영, 비선형성, 확대 투영)를 삽입합니다. LoRA와 달리 병합할 수 없으며 약간의 지연을 추가합니다.
- **프리픽스 튜닝**(prefix tuning)과 **프롬프트 튜닝**(prompt tuning)은 키와 값 또는 입력 임베딩 앞에 붙는 소수의 연속 벡터("가상 토큰")를 학습하고, 모델 전체는 고정해 둡니다.
- **BitFit**은 편향 항만 학습합니다.

LoRA를 위한 실용적인 지침은 다음과 같습니다.

- $r = 8$ 또는 $16$, $\alpha = 2r$로 시작하고, 과제가 사전 학습과 매우 다르면 $r$을 늘립니다.
- $W_Q$와 $W_V$에만 적용하기보다 어텐션과 FFN의 모든 선형 층에 LoRA를 적용합니다.
- 작은 행렬만 학습되고 $B$가 0에서 시작하므로, 전체 미세 조정보다 큰 학습률(예: $10^{-4}$에서 $2 \cdot 10^{-4}$)을 사용합니다.
- 기본 모델이 고정되어 있으므로 LoRA는 전체 미세 조정(31장)보다 덜 잊는 경향이 있지만, 큰 변화가 필요한 과제에서는 덜 배울 수도 있습니다.

## 요약

- 큰 모델의 전체 미세 조정에는 많은 메모리(가중치, 그래디언트, 옵티마이저 상태)와 과제마다 모델의 전체 사본이 필요합니다.
- 미세 조정이 만드는 업데이트는 저랭크 행렬로 잘 근사되는 경우가 많으며, PEFT 방법은 소수의 파라미터만 학습합니다.
- LoRA는 고정된 $W_0$, 0으로 초기화된 $B$, 학습 가능한 $A, B$로 $h = W_0 x + \frac{\alpha}{r} BAx$를 계산합니다. 학습 후에는 업데이트를 병합할 수 있으므로 추론이 느려지지 않습니다.
- QLoRA는 고정된 기본 모델을 이중 양자화를 적용한 4비트 NF4로 저장하고, 16비트 LoRA 어댑터를 학습하며, 페이지드 옵티마이저를 사용하여 GPU 한 대로 매우 큰 모델을 미세 조정할 수 있게 합니다.
- 다른 PEFT 방법으로는 어댑터, 프리픽스 튜닝, 프롬프트 튜닝, BitFit이 있습니다.

## 연습 문제

1. LoRA에서 $B$를 0으로 초기화하는 이유를 자신의 말로 설명하십시오. $A$와 $B$가 모두 무작위라면 무엇이 잘못될까요?
2. 어떤 트랜스포머에 32개의 층이 있습니다. 각 층에서 $4096 \times 4096$ 크기의 어텐션 행렬 네 개에 $r = 16$인 LoRA를 적용합니다. 학습 가능한 파라미터는 모두 몇 개입니까?
3. 130억 개 파라미터 모델의 가중치를 16비트 형식과 4비트 형식으로 저장하는 데 필요한 메모리를 추정하십시오(블록 스케일은 무시).
4. 양자화 코드에서 블록 크기를 64에서 1024로 바꾸고 평균 절대 오차를 비교하십시오. 결과를 설명하십시오.

## 핵심 용어

| 한국어 | English | Русский | 러시아어 발음 |
|---|---|---|---|
| 매개변수 효율적 미세 조정 | parameter-efficient fine-tuning (PEFT) | параметрически эффективное дообучение | parametrícheski effektívnoye doobuchéniye |
| 내재 차원 | intrinsic dimension | внутренняя размерность | vnútrennyaya razmérnost' |
| 랭크(계수) | rank | ранг | rang |
| 저랭크 행렬 | low-rank matrix | матрица низкого ранга | mátritsa nízkogo ránga |
| 저랭크 적응 | low-rank adaptation (LoRA) | низкоранговая адаптация | nizkorángovaya adaptátsiya |
| 어댑터 | adapter | адаптер | adápter |
| 스케일링 계수 | scaling factor | масштабный коэффициент | masshtábnyy koeffitsiyént |
| 가중치 병합 | weight merging | слияние весов | sliyániye vesóv |
| 학습 가능한 파라미터 | trainable parameters | обучаемые параметры | obucháyemye parámetry |
| 양자화 | quantization | квантование | kvantovániye |
| 역양자화 | dequantization | деквантование | dekvantovániye |
| 이중 양자화 | double quantization | двойное квантование | dvoynóye kvantovániye |
| 페이지드 옵티마이저 | paged optimizer | страничный оптимизатор | straníchnyy optimizátor |
| 프리픽스 튜닝 | prefix tuning | префиксная настройка | préfiksnaya nastróyka |
| 프롬프트 튜닝 | prompt tuning | настройка промптов | nastróyka prómptov |

---
[← 이전: 미세 조정과 전이 학습](31-fine-tuning.md) · [목차](README.md) · [다음: 지시 튜닝 →](33-instruction-tuning.md)
