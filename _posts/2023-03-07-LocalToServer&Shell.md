---
title : Local to Server & Shell
categories:
    - Server
date: 2023-03-07
toc: true
toc_label: "Local to Server & Shell"
toc_sticky: True
# toc_icon: ""
sidebar:
  nav: 'counts'
---

# [Terminal] 원격 접속 및 파일 저장

## 1. 원격 Shell 접속

- ssh [원격지서버 UserID]@[원격지서버 HostName]

  - ex)
    서버 IP : 111.222.333.444
    계정명 : abc

  - ```
    ssh abc@111.222.333.444
    ```

- 연결 종료

  - exit 명령어
  - Ctrl+d

## 2. Local → Server

- **scp [옵션] [원본 경로 및 파일] [계정명]@[원격IP주소]:[전송할 경로]**

  - ex)
    서버 IP : 111.222.333.444
    계정명 : abc
    로컬 파일 : /User/home/test.csv
    서버 저장 위치 : /home/abc/Dataset

  - ```
    scp /User/home/test.csv abc@111.222.333.444:/home/abc/Dataset/
    ```

## 3. Server → Local

- **scp [옵션] [원본 경로 및 파일] [계정명]@[원격IP주소]:[원본 경로 및 파일] [전송받을 위치]**

  - ex) 
    서버 IP : 111.222.333.444
    계정명 : abc
    서버에 있는 파일 : /home/abc/test.csv
    로컬 저장 위치 : /User/home/

  - ```
    scp abc@111.222.333.444:/home/abc/test.csv /User/home/
    ```

## 4. ssh포트를 기본 22번으로 사용하고 있지 않는 서버로의 전송

- 예제3) 2222번인 SSH포트를 사용한다면 아래와 같이 -P옵션과 포트번호를 넣어준다.

  - ```
    scp -P 2222 abc@111.222.333.444:/home/abc/test.csv /User/home/
    ```

  - ```
    scp -P 2222 /User/home/test.csv abc@111.222.333.444:/home/abc/
    ```

- 주의사항

  - 옵션 중에 -P와 -p가 있으니 대/소문자 확인
  - -P : 포트번호를 지정함
  - -p : 원본파일 수정/사용시간 및 권한을 유지함
  - -r : 하위 디렉토리 및 파일 모두 복사함

[ 참고 ]

- 원격 Shell 접속
  - https://eunguru.tistory.com/122
- Local to Server & Server to Local
  - https://faq.hostway.co.kr/?mid=Linux_ETC&page=8&document_srl=1426