# 제25장. 현대의 오픈 모델: LLaMA, GQA, Mistral, Qwen

> **제5부 — 실전 언어 모델** · 수준: 중급
>
> **이 장에서 배울 내용:** 대부분의 오픈 디코더 모델이 공유하는 "LLaMA 레시피" · 그룹 쿼리 어텐션이 KV 캐시 메모리를 줄이는 방법 · Mistral의 슬라이딩 윈도 어텐션과 Qwen의 큰 어휘가 레시피에 더하는 것

## 설정 파일 읽기

오픈 모델을 내려받으면 가장 먼저 보게 되는 파일 중 하나가 설정 파일입니다. 전형적인 설정은 대략 다음과 같습니다(숫자는 예시이며 특정 모델의 값이 아닙니다).

```json
{
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 14336,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-05,
  "rope_theta": 10000.0,
  "vocab_size": 32000,
  "max_position_embeddings": 8192,
  "tie_word_embeddings": false
}
```

제16–23장을 거친 지금, 여러분은 대부분을 읽을 수 있습니다. 너비 4096의 층 32개, 어텐션 헤드 32개, 3만 2천 개 토큰의 어휘입니다. 하지만 새로운 필드도 있습니다. 어텐션 헤드가 32개인데 왜 *키–값* 헤드는 8개뿐일까요? 왜 활성화 함수는 `silu`이고 정규화는 `rms_norm`일까요? `rope_theta`는 무엇일까요? 이 필드들은 대부분의 오픈 모델이 공유하는 설계 선택을 나타냅니다. 이 장에서는 그 공통 레시피를 살펴본 뒤, 영향력 있는 세 계열인 LLaMA, Mistral, Qwen이 무엇을 기여했는지 봅니다.

"오픈"이라는 말에 대해 한마디 해 둡니다. 이 모델들 대부분은 **오픈 웨이트 모델**(open-weight model)입니다. 학습된 파라미터는 내려받을 수 있지만 학습 데이터와 전체 학습 코드는 보통 공개되지 않으며, 라이선스도 모델마다 다릅니다. 엄밀한 의미의 완전한 오픈 소스 모델이라면 데이터와 코드도 공개할 것입니다. 모델을 사용하기 전에 항상 라이선스를 읽으십시오.

## LLaMA 레시피

**LLaMA**(Touvron et al., 2023, Meta)는 공개된 데이터만으로 학습한 디코더 전용 모델 계열입니다. 논문은 제23장의 GPT-2 아키텍처에 가한 일련의 수정을 명확히 설명했습니다. 각각의 선택은 새로운 것이 아니었지만, 그 조합이 오픈 모델의 사실상 표준이 되었습니다.

1. **RMSNorm을 이용한 사전 정규화**(제18장). 각 하위 층은 입력을 다음과 같이 정규화합니다.
   $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot g$$
   여기서 $d$는 은닉 차원, $\epsilon$은 수치 안정성을 위한 작은 상수(`rms_norm_eps` 필드), $g$는 학습되는 스케일 벡터입니다. LayerNorm과 비교하면 평균 빼기와 이동이 없어서 계산이 약간 더 저렴하고, 실제로는 똑같이 잘 작동합니다.
2. **SwiGLU 피드포워드 네트워크**(제18장):
   $$\text{FFN}(x) = W_2\big(\text{SiLU}(W_1 x) \odot W_3 x\big)$$
   여기서 $W_1$과 $W_3$는 $d$에서 은닉 크기 $d_{ff}$로 보내고, $W_2$는 다시 되돌리며, $\odot$는 원소별 곱입니다. 행렬이 두 개가 아니라 세 개이므로 $d_{ff}$를 $4d$ 대신 약 $\tfrac{8}{3}d$로 줄여 파라미터 수를 비슷하게 유지합니다. 이것이 `intermediate_size`가 단순히 4의 배수가 아닌 이유입니다.
3. **회전 위치 임베딩, RoPE**(rotary position embedding, 제17장). 입력에 학습된 위치를 더하는 대신 모든 층에서 쿼리와 키에 적용합니다. `rope_theta`는 회전 주파수의 밑이며, 더 긴 문맥을 지원하기 위해 더 큰 값을 씁니다.
4. 대부분의 선형 층에 **편향**(bias)이 **없으며**, SentencePiece BPE 토크나이저(제3장)를 씁니다.

또 하나의 영향력 있는 결정은 학습 길이에 관한 것이었습니다. 제27장에서 공부할 스케일링 법칙 분석은 고정된 *학습* 예산을 모델 크기와 데이터 양 사이에 어떻게 나눌지 알려 줍니다. LLaMA는 그 분석이 권하는 것보다 훨씬 많은 데이터로 비교적 작은 모델을 의도적으로 학습시켰습니다. 그 이유는 더 작은 모델이 *사용할* 때마다 더 저렴하기 때문입니다. 많은 사용자에게 서비스할 모델이라면 추론 비용이 학습 비용만큼 중요합니다.

이후 버전들은 이 핵심을 유지하면서 세부를 바꾸었습니다. 더 긴 문맥, 더 많은 학습 데이터, 더 큰 어휘, 그리고 다음 절의 주제인 그룹 쿼리 어텐션입니다.

## 그룹 쿼리 어텐션

제23장에서 보았듯이 생성 중에 모델은 매 단계 다시 계산하지 않도록 이전 모든 토큰의 키와 값, 즉 **KV 캐시**(KV cache)를 저장합니다(자세한 내용은 제36장). 그 크기는 다음과 같습니다.

$$\text{메모리} = 2 \cdot L \cdot H_{kv} \cdot d_h \cdot T \cdot b$$

여기서 계수 2는 키와 값을 세는 것이고, $L$은 층 수, $H_{kv}$는 키–값 헤드 수, $d_h$는 헤드 차원, $T$는 캐시된 토큰 수, $b$는 숫자 하나당 바이트 수(16비트 부동소수점이면 2)입니다. $L = 32$, $H_{kv} = 32$, $d_h = 128$, $T = 4096$이면 *시퀀스 하나에* 정확히 2 GiB입니다. 긴 문맥으로 많은 사용자에게 서비스하면 GPU가 금세 가득 찹니다. 또한 매 단계 이 캐시를 읽어야 하므로 생성 속도는 계산이 아니라 **메모리 대역폭**(memory bandwidth)에 의해 제한됩니다.

표준 **멀티헤드 어텐션**(multi-head attention, MHA, 제16장)에서는 $H$개의 쿼리 헤드마다 자신의 키 헤드와 값 헤드가 있으므로 $H_{kv} = H$입니다. **멀티쿼리 어텐션**(multi-query attention, MQA; Shazeer, 2019)은 반대 극단으로 갑니다. 모든 쿼리 헤드가 하나의 키 헤드와 하나의 값 헤드를 공유하여 $H_{kv} = 1$입니다. 이렇게 하면 캐시가 $H$배 줄지만 품질이 떨어지고 학습이 덜 안정적일 수 있습니다.

**그룹 쿼리 어텐션**(grouped-query attention, GQA; Ainslie et al., 2023)은 그 절충안입니다. $H$개의 쿼리 헤드를 $H_{kv}$개의 그룹으로 나누고, 각 그룹이 하나의 키 헤드와 하나의 값 헤드를 공유합니다. 쿼리 헤드 32개와 KV 헤드 8개라면 쿼리 헤드 4개마다 같은 키와 값을 공유합니다. GQA는 캐시를 $H / H_{kv} = 4$배 줄이며, 실험에서 품질은 MHA에 가깝게 유지되었습니다. Ainslie et al.은 또한 기존 MHA 체크포인트를 각 그룹 안의 키·값 헤드를 평균한 뒤 잠깐 추가 학습하여 GQA로 바꿀 수 있음을 보였고, 이를 *uptraining*이라고 불렀습니다.

```python
import torch
import torch.nn.functional as F

def gqa(x, Wq, Wk, Wv, n_heads, n_kv_heads):
    B, T, d = x.shape
    hd = d // n_heads                                         # 헤드 차원
    q = (x @ Wq).view(B, T, n_heads, hd).transpose(1, 2)      # (B, H, T, hd)
    k = (x @ Wk).view(B, T, n_kv_heads, hd).transpose(1, 2)   # (B, H_kv, T, hd)
    v = (x @ Wv).view(B, T, n_kv_heads, hd).transpose(1, 2)
    group = n_heads // n_kv_heads                             # K/V 헤드 하나당 쿼리 헤드 수
    k = k.repeat_interleave(group, dim=1)                     # 각 K/V 헤드를 공유
    v = v.repeat_interleave(group, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return out.transpose(1, 2).reshape(B, T, d)

torch.manual_seed(0)
d, H, H_kv = 64, 8, 2
x = torch.randn(1, 5, d)                                      # 토큰 5개
Wq = torch.randn(d, d) / d**0.5
Wk = torch.randn(d, H_kv * d // H) / d**0.5                   # K, V 투영은 4배 작음
Wv = torch.randn(d, H_kv * d // H) / d**0.5
print(gqa(x, Wq, Wk, Wv, H, H_kv).shape)                      # torch.Size([1, 5, 64])

def kv_cache_gib(layers, kv_heads, head_dim, tokens, bytes_per=2):  # 2바이트 = fp16
    return 2 * layers * kv_heads * head_dim * tokens * bytes_per / 2**30  # 2 = K와 V

for kv in (32, 8, 1):                                         # MHA, GQA, MQA
    print(f"{kv:2d} KV heads: {kv_cache_gib(32, kv, 128, 4096):.4f} GiB")
```

출력은 MHA가 2 GiB, KV 헤드 8개인 GQA가 0.5 GiB, MQA가 0.0625 GiB임을 보여 줍니다. `repeat_interleave`는 계산할 때만 존재한다는 점에 주목하십시오. 캐시에 *저장되는* 것은 반복하기 전의 작은 `k`와 `v` 텐서입니다. 키와 값의 투영 행렬도 더 작아지므로 GQA는 파라미터도 조금 절약합니다.

## Mistral: 슬라이딩 윈도 어텐션

**Mistral 7B**(Jiang et al., 2023)는 GQA를 포함한 LLaMA 스타일 레시피를 쓰고 **슬라이딩 윈도 어텐션**(sliding window attention, SWA)을 더했습니다. 일반적인 인과적 어텐션에서 토큰 $i$는 $j \le i$인 모든 토큰에 어텐션합니다. 크기 $W$의 윈도를 쓰면 마지막 $W$개 토큰, 즉 $i - W < j \le i$에만 어텐션합니다. $T = 6$, $W = 3$일 때 마스크는 다음과 같습니다(1 = 허용).

```text
     j: 1 2 3 4 5 6
i = 1:  1 0 0 0 0 0
i = 2:  1 1 0 0 0 0
i = 3:  1 1 1 0 0 0
i = 4:  0 1 1 1 0 0
i = 5:  0 0 1 1 1 0
i = 6:  0 0 0 1 1 1
```

그렇다면 모델은 $W$개 토큰보다 오래된 것을 모두 잊어버릴까요? 꼭 그렇지는 않습니다. 1층에서 토큰 $i$는 최대 $W$개 토큰 뒤까지 봅니다. 하지만 그 토큰들은 자신의 1층 계산에서 이미 $W$개 토큰 더 뒤를 보았습니다. $L$개 층을 지나면 정보는 약 $L \times W$ 위치까지 전달될 수 있습니다. 이미지 신경망에서 합성곱을 쌓으면 깊은 층일수록 더 넓은 영역을 보는 것과 같습니다. 이 범위를 **수용 영역**(receptive field)이라고 합니다.

실질적인 이득은 캐시에 있습니다. 어떤 토큰도 $W$ 위치보다 오래된 키와 값을 필요로 하지 않으므로, 캐시는 고정 크기의 **롤링 버퍼 캐시**(rolling buffer cache)가 될 수 있습니다. 위치 $i$는 슬롯 $i \bmod W$에 저장되어 가장 오래된 항목을 덮어씁니다. 메모리는 더 이상 시퀀스 길이에 따라 늘어나지 않습니다. Mistral 팀은 이후 전문가 혼합 모델인 **Mixtral**도 공개했는데, 이는 제29장에서 공부합니다. 이 모델들은 허용적인 라이선스로 공개되어 널리 퍼지는 데 도움이 되었습니다.

## Qwen과 어휘의 역할

**Qwen**(Bai et al., 2023, Alibaba)은 또 하나의 큰 오픈 모델 계열로, 다양한 크기로 공개되었고 이후 버전에는 코드, 수학, 멀티모달, 전문가 혼합 변형도 있습니다. 아키텍처는 다시 LLaMA 레시피(RMSNorm, SwiGLU, RoPE)를 따르며, 쿼리·키·값 투영에 편향이 있는 것과 같은 작은 차이가 있습니다. 이후 세대는 GQA도 채택했습니다.

가장 눈에 띄는 차이는 **어휘 크기**(vocabulary size)입니다. Qwen은 약 15만 개 토큰의 byte-level BPE 어휘를 쓴다는 점에서 LLaMA와 다릅니다. 원래 LLaMA의 3만 2천 개보다 몇 배 큽니다. 큰 어휘는 절충입니다.

- **장점:** 여러 언어, 특히 중국어뿐 아니라 러시아어와 한국어의 텍스트도 더 적은 토큰으로 나뉩니다. 제3장에서 보았듯이 토크나이저의 **생산성**(tokenizer fertility), 즉 단어당 평균 토큰 수가 중요합니다. 토큰이 적으면 컨텍스트 창에 더 많은 텍스트가 들어가고, 생성 단계가 줄며, 문장당 비용이 낮아집니다.
- **비용:** 임베딩 행렬과 출력 층이 $V \cdot d$로 커집니다. $V = 150{,}000$, $d = 4096$이면 행렬 하나에 약 6억 개의 파라미터이고, 어휘 전체에 대한 소프트맥스도 더 비싸집니다. 또한 드문 토큰은 학습 갱신을 덜 받습니다.

작은 모델에서는 임베딩 행렬이 전체 파라미터에서 큰 비중을 차지할 수 있습니다. 그래서 작은 모델은 입력과 출력 임베딩을 공유하는 경우가 많고(제23장), 큰 모델은 보통 그렇지 않습니다. 어휘와 언어 간 균형은 다국어 모델을 다루는 제30장에서 다시 살펴봅니다.

## 공통된 그림

| 구성 요소 | GPT-2 (제23장) | 전형적인 현대 오픈 모델 |
|---|---|---|
| 정규화 | LayerNorm, 사전 정규화 | RMSNorm, 사전 정규화 |
| 피드포워드 | GELU, $d_{ff} = 4d$ | SwiGLU, $d_{ff} \approx \tfrac{8}{3}d$ 또는 그 이상 |
| 위치 | 학습되는 절대 위치 | RoPE |
| 어텐션 | MHA | GQA, 때때로 슬라이딩 윈도 |
| 편향 | 있음 | 대부분 없음 |
| 어휘 | 약 5만 BPE | 3만 2천 – 15만 이상 BPE |

주된 교훈은 2017년의 트랜스포머가 놀라울 만큼 적게 바뀌었다는 것입니다. 오픈 모델의 발전은 주로 더 나은 데이터, 더 많은 학습, 더 긴 문맥, 그리고 GQA 같은 효율 개선에서 왔습니다. 이는 제6부와 제8부의 주제입니다.

## 요약

- 대부분의 오픈 디코더 모델은 "LLaMA 레시피"를 공유합니다. RMSNorm 사전 정규화, SwiGLU FFN, RoPE, 편향 없음, BPE 토크나이저입니다.
- 오픈 웨이트 모델은 파라미터를 공개하지만 보통 데이터는 공개하지 않으며, 라이선스는 서로 다릅니다.
- KV 캐시 크기는 $2 L H_{kv} d_h T b$입니다. 그룹 쿼리 어텐션은 각 키–값 헤드를 쿼리 헤드 그룹이 공유하게 하여 품질 손실이 거의 없이 캐시를 $H / H_{kv}$배 줄입니다.
- Mistral의 슬라이딩 윈도 어텐션은 각 층이 마지막 $W$개 토큰만 보게 합니다. 층을 쌓으면 여전히 약 $L \times W$의 수용 영역이 생기고, 롤링 버퍼가 캐시 메모리를 일정하게 유지합니다.
- Qwen의 큰 어휘는 많은 언어에서 텍스트당 토큰 수를 줄이지만, 그 대가로 임베딩 행렬과 출력 층이 커집니다.

## 연습 문제

1. GQA가 생성 중의 메모리는 줄이지만 학습 중의 계산량에는 별 영향을 주지 않는 이유를 자신의 말로 설명하십시오.
2. $L = 80$, $H = 64$, $H_{kv} = 8$, $d_h = 128$, 문맥 $T = 8192$, 16비트 숫자, 배치 4개 시퀀스인 모델의 KV 캐시 크기를 계산하십시오. MHA라면 얼마가 필요합니까?
3. `gqa`가 윈도 크기 `W`를 받아 `is_causal=True` 대신 슬라이딩 윈도 어텐션용 불리언 마스크를 명시적으로 만들도록 수정하십시오. `W`가 `T`보다 크면 출력이 인과적 버전과 같은지 확인하십시오.
4. $V = 32{,}000$인 토크나이저는 어떤 러시아어 텍스트를 1,500개 토큰으로, $V = 150{,}000$인 토크나이저는 1,000개 토큰으로 나눕니다. $d = 4096$일 때 더 큰 어휘는 임베딩 파라미터를 몇 개 더 쓰며, 문맥 길이를 몇 퍼센트 절약합니까?

## 핵심 용어

| 한국어 | English | Русский | 러시아어 발음 |
|---|---|---|---|
| 오픈 웨이트 모델 | open-weight model | модель с открытыми весами | modél' s otkrýtymi vesámi |
| 회전 위치 임베딩 | rotary position embedding (RoPE) | ротационное позиционное кодирование | rotatsiónnoye pozitsiónnoye kodírovaniye |
| KV 캐시 | KV cache | KV-кэш | ka-ve-késh |
| 멀티헤드 어텐션 | multi-head attention | многоголовое внимание | mnogogolóvoye vnimániye |
| 멀티쿼리 어텐션 | multi-query attention (MQA) | внимание с общими ключами и значениями | vnimániye s óbshchimi klyuchámi i znachéniyami |
| 그룹 쿼리 어텐션 | grouped-query attention (GQA) | внимание с группировкой запросов | vnimániye s gruppiróvkoy zaprósov |
| 슬라이딩 윈도 어텐션 | sliding window attention | внимание со скользящим окном | vnimániye so skol'zyáshchim oknóm |
| 롤링 버퍼 캐시 | rolling buffer cache | кольцевой буфер кэша | kol'tsevóy búfer késha |
| 수용 영역 | receptive field | рецептивное поле | retseptívnoye póle |
| 어휘 크기 | vocabulary size | размер словаря | razmér slovaryá |
| 토크나이저 생산성 | tokenizer fertility | фертильность токенизатора | fertíl'nost' tokenizátora |
| 메모리 대역폭 | memory bandwidth | пропускная способность памяти | propusknáya sposóbnost' pámyati |
| 편향 | bias | смещение | smeshchéniye |
| 체크포인트 | checkpoint | контрольная точка | kontról'naya tóchka |

---
[← 이전: 텍스트 생성: 탐욕적 디코딩, 빔 서치, 온도, Top-k, Top-p](24-text-generation.md) · [목차](README.md) · [다음: 학습 데이터: 수집, 정제, 중복 제거 →](26-training-data.md)
