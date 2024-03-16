# Linear Model

## 선형성의 의미

선형적으로 증가/감소한다 라는 말은 무슨 의미일까?

> 어떤 값을 상수 배 했을 때 일정하게 커진다.

y = 2x 방정식을 생각해보자.

| x   | y   |
| --- | --- |
| 1   | 2   |
| 2   | 4   |
| 3   | 6   |

똑같은 비율로 증가한다.

Linear Model은 입력 값을 선형적으로 증가/감소 시키는 모델이다.

모델을 $y=wx$로 정의할 수 있다.

그럼 Linear Regression을 할 수 있다.

위의 $y=2x$ 예시를 다시 사용해보자.

아래와 같이 입력이 주어질 때 출력이 정의된다고 하자.

| x   | y   | y’(prediction) |
| --- | --- | -------------- |
| 1   | 2   |
| 2   | 4   |
| 3   | 6   |

우리는 $y=wx$인 상태에서 아래의 입력과 출력을 만족하는 $w$를 찾아야 한다.

그런 $w$만 찾는다면 입력이 어떤 수가 되더라도 출력을 알 수 있기 때문이다!

어떻게 할까?

처음에는 랜덤 값을 준다. $w$가 3이라고 하자.

$y=3x$라서 출력은 아래와 같을 것이다.

| x   | y   | y’(prediction, w=3) |
| --- | --- | ------------------- |
| 1   | 2   | 3                   |
| 2   | 4   | 6                   |
| 3   | 6   | 9                   |

## 손실 구하기

이제 $w=3$일 때 출력값과 실제 정답인 $y$값을 비교해보자.

둘을 비교하는 가장 간단한 방법은 두 값의 차를 구하는 것이다. 이를 손실(loss)이라고 한다.

딥러닝에서 손실은 그냥 두 수의 차를 쓰는 게 아니라 두 수의 차의 제곱을 사용한다.

$loss = (y’-y)^2 = (x*w-y)^2$

| x   | y   | y’(prediction, w=3) | loss(w=3) |
| --- | --- | ------------------- | --------- |
| 1   | 2   | 3                   | 1         |
| 2   | 4   | 6                   | 4         |
| 3   | 6   | 9                   | 9         |

각 입력에 대한 loss를 구하고 모두 더해서 표본 개수로 나누어준다.

이를 MSE(Mean Square Error)라고 한다.

`MSE = 14/3`이 된다.

손실을 줄이는 것이 곧 정답값과 우리가 예측한 값이 가까워지는 것과 같다.

**즉 우리는 손실을 줄여 가장 적절한 $w$를 찾아야만 한다!**