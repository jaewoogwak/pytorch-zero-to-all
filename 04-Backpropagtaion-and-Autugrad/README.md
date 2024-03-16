# Backpropagation and Autograd in Pytorch

## Computing gradient in complicated network

이전 예제에서는 간단한 네트워크에서 gradient를 계산했다.

![alt text](/assets/images/image.png)

그런데 아래처럼 복잡한 네트워크에서도 manualy하게 gradient를 계산하는 것은 어렵다.

![alt text](/assets/images/image-1.png)

Pytorch에서는 계산 그래프(Computational graph)와 연쇄법칙(chain rule)로 gradient를 쉽게 계산할 수 있다.

![alt text](/assets/images/image-2.png){: width="50%" height="50%"}

연쇄법칙에 대해서는 각종 블로그에서 많이들 다루니 패스하고 연쇄법칙을 이용하여 계산 그래프에서 어떻게 손실함수(loss function)에서 가중치(w)에 대한 기울기(gradient)를 구하는지 알아보자.

# Backpropagation

아래와 같이 계산 그래프가 존재한다.

### $loss(w) = s^2 = (\hat{y} - y)^2 = (x*w - y)^2$

![alt text](/assets/images/image-3.png){: width="50%" height="50%"}

첫 번째로 Forward를 통해 손실(loss)값을 구한다.

그리고 Backward propagation(backpropagation)을 통해 $\frac{dloss}{dw}$를 구해준다.

이때 $\frac{dloss}{dw}$는 합성함수의 곱으로 나타나서 연쇄법칙에 의해 계산할 수 있다.

태블릿에 풀어서 작성해보았다.

![alt text](/assets/images/IMG_67666C72F454-1.jpeg){: width="50%" height="50%"}

# Autograd

중요한 것은 Pytorch에서는 위에서 계산 그래프와 연쇄법칙으로 구한 gradient를 자동으로 계산해준다는 것이다.

그래서 일일이 계산할 필요 없이 Pytorch에서 제공하는 tensor 메서드 연산으로 gradient를 계산하면 된다.

먼저 w를 `torch.tensor`를 이용해 텐서로 선언한다.

```python
w = torch.tensor([1.0], requires_grad=True)
```

앞서 언급했듯이 아래 과정으로 훈련이 진행된다.

1. Forward
2. Compute Loss
3. Backward
4. Update weights

여기서 Backward는 `loss.backward()`로 한번에 진행할 수 있다.

그러면 내부에서 자동으로 gradient가 계산된다.

계산한 가중치(w)에 대한 gradient는 `w.grad`로 쉽게 구할 수 있다.

```python
# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val) # 1) Forward pass
        l = loss(y_pred, y_val) # 2) Compute loss
        l.backward() # 3) Back propagation to update weights
        print("\tgrad: ", x_val, y_val, w.grad.item())
        w.data = w.data - 0.01 * w.grad.item() # Update weights

        # Manually zero the gradients after updating weights
        w.grad.data.zero_()

    print(f"Epoch: {epoch} | Loss: {l.item()}")
```

Autograd를 제공하는 Pytorch의 강력함을 느낄 수 있다.
