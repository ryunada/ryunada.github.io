---
title : "Attention & Transformer"
categories:
    - DL
date: 2024-07-23
toc: true
toc_label: "Concept"
toc_sticky: True
# toc_icon: ""
sidebar:
  nav: 'counts'
---

<span style = 'color:blue'>**By Seung Ho Ryu (Comment. Jung In Seo)**</span>
<details>
    <summary>Reference</summary>

- 파이썬 텍스트 마이닝 완벽 가이드(Section 13)
- 텐서플로 2와 머신러닝으로 시작하는 자연어 처리(Section 6)
- 케라스 창시자에게 배우는 딥러닝(2판)(Section 11.5)

</details>

# 1. Seq2Seq(Sequence to Sequence)

- 시퀀스 형태의 입력값을 시퀀스 형태의 출력으로 만들 수 있게 하는 모델  
    - `Sequence` : 단어, 문자, 또는 기타 연속된 데이터 포인트들의 집합  
- 인코더(Encoder)와 디코더(Decoder)를 주요 구성요소로 가지고 있음
    - ex) "hello world"에서 "bonjour le monde"로 번역하는 과정
        - \<start> : 문장의 시작 혹은 번역의 시작을 알리는 토큰
        - \<end> : 문장의 마지막을 알리는 토큰
    - <img src = "/assets/img/DL/Transformer/Seq2Seq.png" width = "100%" alt = "Seq2Seq">
    - **Encoder**
        - 모든 단어를 순차적으로 입력 받은 뒤 마지막에 이 모든 단어들의 정보를 압축해서 담은 `Context Vector`생성
        - `Context Vector` : 인코더 RNN셀의 마지막 시점의 은닉 상태(인코더 부분의 정보를 요약해 담고 있는 벡터)
    - **Decoder**          
        - Context Vector와 \<start>를 받아 순차적으로 토큰을 만들어 최종 출력 시퀀스를 만들어냄
        - [ 훈련 과정 ]
            - Encoder의 입력에 대응하는 실제값을 입력으로 사용
            - 디코더의 입력을 이전 시점의 은닉상태와 현재 시점의 실제 값으로 사용(교사강요)
            - 훈련 과정에서 이전 시점의 예측 값을 사용하다가 예측이 잘못되면 Decoder 전체 예측이 잘못될 수도 있고, 훈련 시간이 느려지기 때문입니다.
        - [ 실제 번역 과정 ]
            - 디코더의 입력이 이전 시점의 은닉상태와 이전 시점의 예측 값

- 가장 많이 활용되는 분야
    - 기계 번역(Machine translation) : 입력 시퀀스를 입력 문장, 출력 시퀀스를 번역 문장으로 구성
    - 챗봇(Chatbot) : 입력 시퀀스를 질문, 출력 시퀀스를 대답으로 구성

- 단점
    - RNN의 기울기 소실 문제
    - 하나의 고정된 크기의 벡터에 모든 정보를 압축하려고 하여 정보 손실이 발생
        - EX) "I went to school yesterday" → "나는 어제 학교에 갔다"
            - 마지막 단어인 '갔다'를 예측할 때, 이 단어에 가장 많은 영향을 미치는 원문의 단어는 'went'이다. 그러나 이 정보는 다른 단어들과 함께 문맥 벡터에 숨어있다.
            - 'went'가 '갔다'의 예측에 직접 관여하기 위하여 고안된 것이 Attention 메커니즘이다.