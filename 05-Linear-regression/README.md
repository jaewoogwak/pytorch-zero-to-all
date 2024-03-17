# Linear Regression

이제 Pytorch에서 어떻게 모델을 선언하고 훈련시키는지 알아보자.

## 1. Model class in pytorch way

먼저 아래와 같이 모델을 정의한다. 모델을 정의할 때는 `nn.Module`을 상속받아야 한다.

```python
class Model(nn.Module):
    ...
```

모델 내부에는 두 가지 메서드가 필수적으로 선언되어야 한다.

1. \_\_init(self)

-   내부엔 레이어를 선언해준다 (ex. `torch.nn.Linear`)

2. forward()

-   앞서 선언한 레이어에 입력(x)을 넣고 예측값(y_pred)을 출력한다

```python
class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1) # One in (x_data) and on out (y_data)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data ad we must return a Variable of output data.
        We can use Modules defined in the constructor as well as arbitrary operators on Variables.
        """

        y_pred = self.linear(x)
        return y_pred

```

## 2. Construct loss and optimizer

모델을 선언했으면 모델의 성능을 평가할 손실 함수를 정의한다.

그리고 훈련 과정에서 모델을 쉽게 사용할 수 있게 도와주는 옵티마이저를 정의한다.

이 예제에서는 `torch.nn.MSELoss`로 손실 함수를 정의하고
`torch.optim.SGD`로 옵티마이저를 정의한다. 이때 SGD는 2개의 파라미터를 받는다.

-   `model.parameters()`로 내가 선언한 모델의 학습파라미터들
-   learning rate

```python
# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction="sum") # output이 loss값의 합이 됨
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

## 3. Training: forward, loss, backward, step

기존에는 아래와 같이 하드코딩해서 일일히 가중치를 업데이트했다.

```python
w.data = w.data - 0.01 * w.grad.data
```

이젠 Pytorch가 제공하는 `optimizer.step()` 메서드로 한번에 가중치 업데이트가 가능하다.

```python
# Training loop
for epoch in range(500):
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # 2) Compute and print loss
    loss = criterion(y_pred, y_data)
    print(f"Epoch: {epoch} | Loss: {loss.item()} ")

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 4. Tesing

모델 훈련이 끝났으니 이제 새로운 데이터로 테스트할 차례이다.

새로운 데이터로 다시 forward해서 결과를 얻어본다.

```python
# After training
hour_var = tensor([[4.0]])
y_pred = model(hour_var)
print("Prediction (after training)", 4, model(hour_var).data[0][0].item())
```
