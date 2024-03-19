# Wide and Deep

아래는 Sung Kim님이 계신 홍콩과기대 학생들의 성적표이다.

<img src="/assets/images/HKUST_PA.png" width="500" height="500">

입력값인 그들의 GPA로 학생들의 가입 여부(0 or 1)를 예측해야한다.

그런데 GPA만으로 가입 여부를 예측할 수 있을까?

현실 세계에서는 다른 요소도 종합적으로 고려하여 평가할 것이다.

그렇기 때문에 이전까지 예제에서는 Linear model에 입력이 1개였지만 (n x 1) 이제 입력 개수가 여러 개인 경우를 살펴보자.

<img src="/assets/images/HKUST_PA2.png" width="500" height="500">

이제 Experience가 추가되어 입력은 N\*2가 되었다.

## Matrix Multiplication

입력이 늘어났을 때 $xw+b = \hat{y}$에 따라 $w$의 크기도 변해야한다.

$x$의 크기는 $N\times2$이고 출력 $\hat{y}$는 $N\times1$이다.

따라서 $w$의 크기도 맞춰줘야 한다.

<img src="/assets/images/matrix_mul.png" width="500" height="500">

행렬 곱을 할 때 $(N\times2) \times (?\times?) = \ (N\times1)$일 경우 각 크기의 내부 요소들끼리 맞춰주어야 한다.

위 예시의 경우 아래처럼 내부 요소(2)가 같아야 행렬 곱이 가능하다.

$(N\times2) \times (2\times1) = \ (N\times1)$

## Go Wide and Deep

결국 현실 세계 문제를 해결하기 위해서 신경망의 입력이 늘어나야 하며(wider) 신경망을 계층구조로 깊게 쌓아야 (deeper) 한다.

## Vanishing Gradient Problem

그러면 아래와 같은 생각을 할 수 있다.

> 신경망을 깊게 쌓을수록 좋지 않을까?

그런데 실제로는 신경망을 깊게 쌓을 경우 훈련이 잘 되지 않는다는 것이 밝혀졌다.

그 문제를 Vanishing Gradient(기울기 소실)이라고 한다.

<img src="/assets/images/vgp.png" width="500" height="500">

Vanishing Gradient의 원인은 Acivation function(활성화 함수)의 기울기와 관련이 있다.

<img src="/assets/images/sigmoid_gradient.png" width="500" height="500">

Sigmoid 함수는 출력 범위가 0~1이다. 그래프 개형을 보면 0에서 기울기가 가장 큰데 이마저도 0.25라는 작은 값이다.

심지어 0에서 멀어질수록 기울기가 0에 수렴하게 된다.

Sigmoid와 같은 Activation function을 이용하여 신경망을 깊게 쌓을 경우 Backward propagation 과정에서 Sigmoid의 기울기(gradient)가 계속해서 곱해지게 된다.

작은 값들이 연속해서 곱해지면서 기울기가 0에 수렴해버리게 된다. 이렇게 되면 앞서 보았던 가중치를 업데이트하는 수식에 영향을 주게 된다.

### $w = w - \alpha * \frac{dloss}{dw}$

위에서 $\frac{dloss}{dw}$가 0에 수렴한다면 $w$도 거의 변하지 않게된다.

따라서 모델이 훈련하더라도 가중치 업데이트 속도가 매우 느려 학습이 거의 되지 않는 문제가 발생한다.

이를 해결하기 위해 ReLU Activation function이 등장한다. 그러나 이 역시 Dying ReLU 현상으로 한계가 있으며 이를 극복하기 위해 Leaky ReLU를 사용한다.
