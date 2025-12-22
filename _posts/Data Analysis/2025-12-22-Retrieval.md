---
title : "Retrieval"
categories:
    - Data Analysis
date: 2025-12-22
toc: true
toc_label: "Concept"
toc_sticky: True
# toc_icon: ""
sidebar:
  nav: 'counts'
math: true           # 수식 설정
---

# RAG에서 Retrieval이란?

RAG(Retrieval-Augmented Generation)는 **외부 지식을 검색(Retrieval)** 하고, 그 결과를 바탕으로 **LLM이 답변을 생성(Generation)** 하는 구조의 아키텍처이다. 이 글에서는 RAG의 핵심 구성요소 중 하나인 **Retrieval 단계**를 중심으로 개념부터 구현 관점까지 정리한다.

---

## 1. 왜 Retrieval이 필요한가?

LLM은 학습 시점 이후의 정보나 특정 도메인의 상세 지식을 정확히 알지 못한다. 또한, 환각(Hallucination) 문제가 존재한다. Retrieval은 이러한 한계를 보완한다.

* 최신 정보 반영 가능
* 도메인 특화 지식 활용
* 근거 기반 응답 생성 → 환각 감소

즉, Retrieval은 **LLM의 기억을 외부로 확장하는 역할**을 한다.

---

## 2. RAG 전체 구조에서 Retrieval의 위치

RAG는 크게 두 단계로 나뉜다.

```
User Query
   ↓
[ Retrieval ]  → 관련 문서 Top-K
   ↓
[ Generation ] → 문서 + Query 기반 답변 생성
```

Retrieval 단계는 **"어떤 정보를 LLM에게 보여줄 것인가"** 를 결정하는 매우 중요한 과정이다.

---

## 3. Retrieval의 핵심 구성 요소

### 3.1 Document (지식 원천)

Retrieval의 대상이 되는 데이터이다.

* PDF, TXT, CSV
* 웹 문서
* DB 레코드
* 로그 데이터

보통 긴 문서는 그대로 쓰지 않고 **Chunking** 과정을 거쳐 작은 단위로 나눈다.

---

### 3.2 Chunking (문서 분할)

문서를 일정 길이로 나누는 과정이다.

* Chunk size (예: 256 ~ 1024 tokens)
* Overlap (문맥 유지를 위해 일부 중첩)

Chunking이 중요한 이유:

* 너무 크면 → 검색 정확도 하락
* 너무 작으면 → 문맥 손실

---

### 3.3 Embedding (벡터화)

각 Chunk를 **고정 길이의 벡터**로 변환한다.

* Sentence-BERT
* OpenAI Embedding
* HuggingFace embedding models

Embedding의 목적은 **의미 기반 검색(Semantic Search)** 을 가능하게 하는 것이다.

---

### 3.4 Vector Store (벡터 저장소)

Embedding된 벡터를 저장하고 검색하는 시스템이다.

* FAISS
* Chroma
* Milvus
* Pinecone

Vector Store는 보통 다음을 포함한다.

* 벡터 값
* 원본 텍스트
* 메타데이터 (출처, 날짜, 카테고리 등)

---

### 3.5 Query Encoder

사용자의 질문(Query)을 벡터로 변환한다.

* Document embedding과 **동일한 모델** 사용
* 그래야 의미 공간이 일치함

```text
Query → Embedding Vector
```

---

### 3.6 Similarity Search (유사도 검색)

Query 벡터와 Document 벡터 간의 유사도를 계산한다.

주요 방식:

* Cosine Similarity
* Inner Product (MIPS)
* L2 Distance

이 중 **MIPS (Maximum Inner Product Search)** 가 대규모 벡터 검색에서 자주 사용된다.

---

## 4. Retrieval 전략

### 4.1 Top-K Retrieval

가장 기본적인 방식이다.

* Query와 가장 유사한 문서 K개 반환
* K 값이 작으면 정보 부족
* K 값이 크면 노이즈 증가

---

### 4.2 Metadata Filtering

메타데이터 조건을 함께 사용한다.

예시:

* 날짜 필터
* 문서 타입 필터
* 프로젝트 ID 필터

```text
Query + Metadata Filter → Relevant Docs
```

---

### 4.3 Hybrid Search

* Keyword Search (BM25)
* Semantic Search (Embedding)

두 방식을 결합하여 장점을 취한다.

---

### 4.4 Re-ranking

1차 Retrieval 결과를 대상으로
2차 모델로 다시 정렬한다.

* Cross-Encoder
* LLM 기반 평가

정확도는 올라가지만 비용이 증가한다.

---

## 5. Retrieval 품질이 RAG 성능에 미치는 영향

RAG 성능의 상당 부분은 Retrieval이 결정한다.

* ❌ 잘못된 문서 → 정답 생성 불가
* ❌ 불충분한 문서 → 추론 실패
* ✅ 정확한 문서 → LLM 성능 극대화

> **Garbage In, Garbage Out**

---

## 6. 실무에서 자주 겪는 문제

* Query와 Document 표현 불일치
* Chunking 전략 미흡
* K 값 튜닝 실패
* 메타데이터 설계 부족

RAG 튜닝의 대부분은 **Retrieval 개선 작업**이다.

---

## 7. 정리

Retrieval은 단순한 검색 단계가 아니라,

* 어떤 지식을
* 어떤 기준으로
* 얼마나 보여줄지

를 결정하는 **RAG의 핵심 설계 요소**이다.

다음 글에서는 Retrieval 결과를 활용하는 **Generation 단계와 Prompt 설계**를 정리할 예정이다.

---

## 참고 키워드

* RAG
* Semantic Search
* Embedding
* Vector Database
* MIPS
