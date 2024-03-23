# Advanced CNN

## Inception model

이전 강의에서 CNN을 배웠다.

CNN의 필터 크기가 1x1, 3x3, 5x5.. 등 다양하다.

<img src="/assets/images/inception_why.png" width="500" height="500">

> 이 필터 크기를 어떻게 정해야 모델이 좋은 성능을 낼까?

그렇게 1x1 필터를 사용하는 실험을 해보게 된다.

<img src="/assets/images/inception_modules.png" width="500" height="500">

## Why using 1x1 filter?

결과적으로 1x1 필터를 사용했을 때 좋은 성능이 나왔다.

왜 그랬을까? 아래와 같은 의문이 들 수 있다.

> Inception module처럼 1x1 filter이후 5x5 or 3x3 filter를 거치는 것과 바로 5x5 filter를 거치는 것을 비교하자. 그럼 후자가 더 낫지 않나? 전자(Inception module)는 두 번의 Convolution이 필요한데?

<img src="/assets/images/why_1x1_convolution" width="500" height="500">

근데 1x1을 추가하면 월등한 operation 감소를 보였다. (약 10배)

결국 1x1 filter를 거친 뒤 3x3 or 5x5 filter를 거치면 연산 수가 매우 크게 줄어들어 학습에 도움이 된다는 것을 알게 됨.

## Go deeper?

Inception model v3/v4의 구현을 보면 레이어가 매우 깊게 쌓여있다.

<img src="/assets/images/deeper" width="500" height="500">

> 그럼 더 깊게 만들수록 더 좋은 거 아닌가?

아니다. 레이어 깊은게 오히려 error가 높다.

<img src="/assets/images/deeper_bad" width="500" height="500">

지나치게 깊은 네트워크는 높은 에러를 보였으며 이것은 많은 데이터셋에서 관측된 일반적인 현상이었다.

그럼 왜 깊은 레이어를 쌓아 네트워크를 구성하는게 더 높은 에러를 보일까?

-   Vanishing Gradient Promblem으로 인해 가중치 업데이트가 잘 되지 않는다.
-   그리고 Backpropagation하면서 Gradient 계산 횟수도 늘어나서 연산 횟수가 크게 증가함.
-   Degradation problem
    -   딥러닝 모델 레이어가 깊어졌을 때 모델이 수렴했음에도 불구하고 오히려 레이어 개수가 적을 때보다 error가 더 커지는 현상 발생
    -   오버피팅 때문이 아니라, 네트워크 구조상 레이어를 깊이 쌓았을 때 최적화가 잘 안 되기 때문 ->
    -   해결책은 Residual block (ResNet)
