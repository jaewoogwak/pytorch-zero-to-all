# Pytorch DataLoader

## Manual data feed

이전 예제에서는 Diabetes dataset을 직접 다운로드하여 모델을 학습시켜 보았다.

Dataset을 다운로드 하고 훈련하는 과정은 아래와 같았다.

```python
xy = np.loadtxt('../data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = from_numpy(xy[:, 0:-1])
y_data = from_numpy(xy[:, [-1]])
print(f'X\'s shape: {x_data.shape} | Y\'s shape: {y_data.shape}')

# Training loop
for epoch in range(100):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch+1}/100 | Loss:{loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights
    optimzier.zero_grad()
    loss.backward()
    optimzier.step()
```

그런데 Dataset 크기가 커지게 되면 일일히 데이터를 모델이 전달하기 힘들다.

또한 모든 데이터의 gradient를 계산하는 것도 효율적이지 않다.

## Batch (Batch size)

그래서 Batch(배치)라는 개념이 등장한다.

훈련 데이터를 배치 단위로 묶어서 활용한다.

신경망 용어에서 훈련 데이터 전체를 1번 순회하는 걸 1 epoch으로 정의한다.

그리고 배치 단위로 나누어서 1 epoch이 될 때까지 순회하는 것이다.

예를 들어 1,000개의 훈련 데이터가 존재하고 Batch size가 500이라면 1 epoch를 만족하기 위해 2 iteration이 필요하다.

## Dataloader

Pytorch에서는 이러한 Batch 개념을 활용하여 Dataloader를 제공한다.

<img src="/assets/images/dataloader.png" width="500" height="500">

우리가 원본 데이터를 무작위로 섞어서 어떻게 나눌지를 고민할 필요 없이 iterable 객체에 넣어 Dataloader에 전달하기만 하면 알아서 Batch 단위로 나누어 준다.

이러한 Dataloader는 아래와 같이 custom하게 정의할 수 있다.

```python
class DiabetesDataset(Dataset):
    """ Diabetes dataset. """
    # Initialize your data, download, etc.
    def __init__(self):
        ...

    def __getItem__(self, index):
        ...

    def __len__(self):
        ...
```

`__init__` 부분에선 데이터를 다운로드하고 읽기를 수행한다.

`__getItem__`은 인덱스에 대한 아이템을 반환하고

`__len__`은 데이터의 길이를 반환한다.
