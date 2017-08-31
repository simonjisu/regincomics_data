레진코믹스 테이터 챌린지 2017
===
레진코믹스 데이터 챌린지 2017에서 제공하는 65만 건의 데이터을 사용해서 1시간 이내로 구매하는 지 예측하는 모델을 만들었습니다. 

결과 노트북은 아래 링크로, 모델확인은 repository안에 있는 py파일들을 확인하시면 됩니다.

예측값은 'test_y_(모델명).txt' 으로 저정해두었습니다.

## 변경사항:
8.31 -  Keras로 예측한 모델을 추가했습니다.

### 8.31: Add New version Jupyter notebook link:
[Link to Jupyter nbviewer](https://nbviewer.jupyter.org/github/simonjisu/regincomics_data/blob/master/Regin%20Comics%20Data%20Analysis_ver2.ipynb)

### Jupyter notebook link:
[Link to Jupyter nbviewer](https://nbviewer.jupyter.org/github/simonjisu/regincomics_data/blob/master/Regin%20Comics%20Data%20Analysis.ipynb)

## 과정:

### 전처리:
만화에 대한 정보와 구매자에 대한 정보가 섞여 있는데, 구매자에 관한 정보(주요 작품 구매 여부) 일부를 구매자 성향으로 바꿔보는 시도를 해보았습니다.

### 방법:
Random Forest를 활용한 구매여부 예측

### 이유:

Random Forest는 Decision Tree를 개별적으로 적용하는 앙상블 방법입니다. 

이 방법을 사용한 이유는 첫째로, Decision Tree방법 자체가 사람이 쉽게 해석할 수 있기 때문입니다. 

둘째로, 독립 변수 차원을 랜덤하게 감소시킨 다음 그 중에서 독립 변수를 선택하기 때문에, 개별 모형들 사이의 상관관계가 줄어들어 모형 성능의 변동이 감소하는 효과가 있기 때문입니다. 

셋째로, 어떤 독립변수가 분류시 중요하게 작용했는지 알 수 있어서 빠르게 예측을 적용할 수 있는 방법이기 때문입니다.

### 결과:

검증 성능을 측정하기 위해 KFold로 9:1 비율의 train과 test set으로 나눠서 검증한 결과 73.25%의 정확도를 나타냈습니다.

구매여부에 대한 예측은 주로 사용자에 대한 정보가 중요하다는 것을 알 수 있었습니다.

구매자 성향인 user_tendency와 purchasing power가 상위권을 이루고 있고, 그 뒤로 만화의 특성이 순위에 올라와있는 모습이었습니다.

### 향후계획:
* 정확도를 더 높일 수 있는 모델을 개발
* 개인성향(user)와 만화(comic)특성을 score화 시켜서 곱한뒤 일종의 평점을 만들어내서, user-comic matrix로 만화 추천 시스템을 만들어 볼 것
