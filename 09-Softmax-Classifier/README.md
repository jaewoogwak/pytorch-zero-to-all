# Softmax Classifier

Pytorch에서는 `nn.CrossEntropyLoss`를 제공한다.

여기에는 Softmax가 내부적으로 구현되어 있다.

따라서 `nn.CrossEntropyLoss`를 호출할 때 y_pred와 y(label)을 그대로 넣기만 하면 된다. 매우 간단하다.

```python
# Cross Entropy Example
# One hot
# 0: 1 0 0
# 1: 0 1 0
# 2: 0 0 1

Y = np.array([1,0,0])
Y_pred1 = np.array([0.7, 0.2, 0.1])
Y_pred2 = np.array([0.1, 0.3, 0.6])
print(f"Loss 1: {np.sum(-Y * np.log(Y_pred1)):.4f}")
print(f"Loss 2: {np.sum(-Y * np.log(Y_pred2)):.4f}")

# Expected
# Loss 1: 0.3567
# Loss 2: 2.3026

```

분류할 것이 여러 개더라도 그대로 하면된다. (Batch loss)

```python
# target is of size nBatch
# each element in target has to have 0 <= value < nClasses (0-2)
# Input is class, not one-hot
Y = tensor([2, 0, 1], requires_grad=False)

# input is of size nBatch x nClasses = 2 x 4
# Y_pred are logits (not softmax)
Y_pred1 = tensor([[0.1, 0.2, 0.9],
                  [1.1, 0.1, 0.2],
                  [0.2, 2.1, 0.1]])
Y_pred2 = tensor([[0.8, 0.2, 0.3],
                  [0.2, 0.3, 0.5],
                  [0.2, 0.2, 0.5]])

                  l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print(f"Batch Loss1: {l1.item():.4f} \nBatch Loss2: {l2.data:.4f}")

# Expected
# Batch Loss1: 0.4966
# Batch Loss2: 1.2389
```

[실습 코드](https://github.com/jaewoogwak/pytorch-zero-to-all/blob/master/09-Softmax-Classifier/softmax_mnist.ipynb)

### 🚀 Exercise

자연어처리및실습 수업 시간에 배운 [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/)을 실습 코드에 적용시켜보았다.

[PL 코드](https://github.com/jaewoogwak/pytorch-zero-to-all/blob/master/09-Softmax-Classifier/sofmax_mnist_with_pl.ipynb)

### Reference

-   https://baeseongsu.github.io/posts/pytorch-lightning-introduction/

-   https://lightning.ai/docs/pytorch/stable/starter/introduction.html

-   https://github.com/jaewoogwak/nlp/blob/master/ch02/with_pl.ipynb
