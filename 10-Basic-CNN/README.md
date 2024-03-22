# Basic CNN

이미지 출처: https://talkingaboutme.tistory.com/entry/DL-Convolution%EC%9D%98-%EC%A0%95%EC%9D%98

## Convolutions

Convolutions(컨볼루션)이란 뭘까?

<img src="/assets/images/convolution.png" width="500" height="500">

컨볼루션에 대해 쉽게 설명한 [블로그](https://talkingaboutme.tistory.com/entry/DL-Convolution%EC%9D%98-%EC%A0%95%EC%9D%98)에 따르면,

아래와 같이 어떤 이미지와 해당 이미지보다는 작은 필터(filter)가 존재한다.

<img src="/assets/images/image_and_filter.png" width="500" height="500">

해당 필터가 이미지를 훑으면서 특징을 추출하고 나면 기존 이미지 크기보다 작은 이미지가 생성된다.

<img src="/assets/images/image_and_filter2.png" width="500" height="500">

이렇게 이미지의 특징(feature)을 추출하는 과정을 컨볼루션이라고 한다.

컨볼루션을 한 뒤 생성되는 결과물을 Feature map이라고 한다.

컨볼루션 과정을 잘 표시한 [블로그](https://github.com/vdumoulin/conv_arithmetic?tab=readme-ov-file)를 참고하자.

## Stride and Padding

필터가 이미지를 훑으면서 feature를 포착한 결과물이 feature map이다. 이런 과정에서 이미지 크기가 줄어들게 된다. 물론 feature map에는 이미지의 특징이 포착되어 있겠지만 이미지가 줄어드는게 싫다면 Stride와 Padding을 적절히 사용해야한다.

<img src="https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/padding_strides.gif" width="500" height="500">

Stride는 필터가 이미지를 훑을 때 얼마나 큰 보폭으로 지나갈지를 결정한다.

Stride가 1이면 한 칸씩 지나가고 2라면 두 칸씩 지나간다.

Padding은 기존 이미지 주변에 Padding을 줌으로서 이미지 크기가 줄어들지 않도록 해주는 것이다.

이제 아래 CNN을 구현해보자.

<img src="/assets/images/cnn.png" width="500" height="500">
