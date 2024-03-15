# Gradient Descent

## Learning의 정의

학습(learning)이란 무엇일까?

> 손실을 최소화시키는 w값을 찾는 것이다.

그러면 손실을 최소화시키는 w값을 어떻게 찾을까?

이러한 손실 함수가 있다.

<img width="855" alt="image" src="https://github.com/jaewoogwak/pytorch-zero-to-all/assets/62415600/a4e0c6b2-eb7b-4a7c-a8fd-8c3a66f3b3ce">

손실 함수 그래프 개형을 보자

초기 $w$(Initial Weight)에서 기울기가 양수이다.

이 기울기로 무엇을 할 수 있을까?

그래프의 개형에 따라 **기울기가 0이 되는 방향으로 가면 손실함수의 최소값에 도달할 수 있다 **

기울기가 양수이면 기울기가 줄어드는 방향으로 움직이고, 기울기가 음수이면 기울기가 커지는 방향으로 움직이다보면 손실함수의 최소값에 도달할 수 있다.

다음과 같이 식으로 표현할 수도 있다.

### $w = w - \alpha * \frac{dloss}{dw}$

여기서 $\alpha$는 $w$를 변화시킬 때 얼마나 크게 변화시킬 것인지에 대한 값이다.

이를 **Learning rate**라고 한다.

Learning rate는 최솟값을 찾아 나설 때 성큼성큼 걸을지, 조금씩 걸어갈지의 보폭과 같다.

보편적으로 0.01, 0.001을 사용한다.

Gradient descent는 아래와 같은 식으로 최소 손실값을 가지는 $w$를 찾는 과정이다!

### $w = w - \alpha * \frac{dloss}{dw}$

이제 구현해보자!
