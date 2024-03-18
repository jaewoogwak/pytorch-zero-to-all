# Logistic regression

## Binary prediction (0 or 1) is very useful!

-   Spent N hours for study, pass or fali?
-   Soccer game against Japan, win or lose?
-   She/he looks good, propse or not?

실세계에서 0 or 1을 출력하는 모델은 많은 곳에 필요하다.

결국 기존처럼 예측 값을 내놓는 Linear model이 아닌 0 or 1을 출력하는 모델도 필요하다.

기존 Linear model에 Sigmoid 함수를 함께 사용하여 출력을 조절할 수 있다.

Sigmoid는 값을 0과 1 사이로 Squash한다.

<img src="/assets/images/sigmoid.png" width="500" height="500">

그럼 Sigmoid 함수로 True or False, 1 or 0를 어떻게 예측할까?

Threshold value를 도입한다.

Threshold value보다 크면 True 아니면 False를 출력하면 된다.

Sigmoid 수식에의해 출력은 아래와 같은 수식이 된다.

### $\hat{y} = \alpha(xw + b)$

## Cross Entropy Loss

그런데 logistic regression에서 (0 or 1 예측) MSELoss가 제대로 작동하지 않는다.

그래서 Cross Entropy Loss 함수를 도입한다.

보편적으로 Regression에는 MSELoss를 사용하고, Classification에는 Cross Entropy Loss를 사용한다.

label과 prediction 값에 따른 Cross Entropy Loss 결과를 케이스별로 알아보자.

<img src="/assets/images/BCELoss_case.png" width="500" height="500">

결과적으로 Correct일 경우 (y와 y_pred의 차이가 거의 없을 경우) loss가 낮고, Wrong일 경우 (y와 y_pred의 차이가 많이 날 경우) loss가 높다.

Sigmoid는 그럼 어케쓸까?

`torch.nn.functional` 패키지를 임포트해서 쓰면 된다.

여기에 Sigmoid 함수를 비롯한 다양한 Activation functions가 존재한다.

> `BCELoss`는 Binary Cross Entropy Loss이다!

## 다시 정리해보는 pytorch식 코드 작성 순서

1. Design your model using class
2. Construct loss and optimizer (select from Pytorch API)
3. Training cycle (forward, backward, update)

---

Image ref: https://iphfly1030.tistory.com/134
