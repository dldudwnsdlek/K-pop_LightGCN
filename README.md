# K-pop 멤버-그룹 관계 지식 그래프를 융합한 LightGCN 기반 음악 추천 시스템

**2025 한국디지털콘텐츠학회 추계종합학술대회 및 대학생논문경진대회 발표 논문**

> 이영준 (강남대학교 경영학과) · 지도교수: 곽찬희 (강남대학교 인공지능융합공학부)

---

## 연구 개요

K-pop은 전 세계적으로 급성장하며 아티스트와 팬 간의 강한 유대 관계를 중심으로 독특한 문화 생태계를 형성하고 있다. 그러나 기존 협업 필터링 기반 추천 시스템은 **콜드 스타트 문제**를 갖고 있어, 예를 들어 BTS 멤버 정국이 솔로 앨범을 발매했을 때 그룹 팬이 멤버의 솔로 활동에도 관심을 가질 것이라는 사실을 반영하지 못한다.

본 연구는 **K-pop 아티스트의 그룹-멤버 관계를 지식 그래프로 명시적으로 모델링**하고, 이를 LightGCN 기반 음악 추천 시스템에 융합하여 추천 정확도를 높이는 것을 목표로 한다.

---

## 모델 구조

```
[Kaggle K-pop 데이터]
        |
        v
[Step 1] 지식 그래프 구축
  - 1,612개 노드 (아티스트 + 그룹)
  - 1,342개 멤버-그룹 엣지
  - 별칭(alias) 매핑 포함
        |
        v
[Step 2] GCN 임베딩 학습
  - 멤버-그룹 관계로부터 "그룹 정체성 임베딩" 학습
  - 멤버 임베딩에 소속 그룹 정보 자연스럽게 반영
        |
        v
[Step 3] 아티스트명 정규화 및 매칭
  - 멜론(한국어) <-> Kaggle(영어) 아티스트명 매칭
  - 18,912곡의 아티스트 정보 성공적으로 매핑
        |
        v
[Step 4] K-pop 비율 기반 데이터 분할
  - 38,531개 플레이리스트 학습 데이터 선정
  - K-pop 비율에 따른 층화 샘플링
  - 플레이리스트 내 70:15:15 분할 (Train:Val:Test)
        |
        v
[Step 5] Fused LightGCN 학습
  - GCN 임베딩을 LightGCN 초기 임베딩 테이블에 가중합으로 주입
  - BPR Loss로 랭킹 학습
  - 3-layer propagation, embedding dim=64
```

---

## 핵심 아이디어: 그룹 정체성 임베딩 융합

기존 접근법과의 차별화:

| 방식 | 설명 |
|------|------|
| KGAT (통합형) | 지식 그래프와 CF를 하나의 그래프로 통합 학습 |
| KLGCN (병렬형) | 지식 그래프와 CF를 병렬로 학습 후 합산 |
| **본 연구 (순차형)** | **GCN으로 그룹 정체성 임베딩을 사전 학습한 뒤 LightGCN 초기 임베딩에 주입** |

K-pop 곡의 아티스트가 지식 그래프에 존재할 경우, 아티스트 임베딩과 소속 그룹 임베딩을 가중합으로 결합하여 LightGCN의 초기 곡 임베딩으로 사용한다.

---

## 데이터셋

| 데이터 | 설명 |
|--------|------|
| [Melon Playlist Dataset](https://github.com/kakaoenterprise/melon-playlist-dataset) | 멜론 플레이리스트 및 곡 메타데이터 (ICASSP 2021) |
| Kaggle K-pop Idol Dataset | K-pop 아이돌 정보 (Stage Name, Korean Name, Group 등) |

**전처리 결과:**
- 지식 그래프: 1,612 노드 / 1,342 멤버-그룹 엣지
- 매핑된 K-pop 곡: 18,912곡
- 학습 플레이리스트: 38,531개
- 데이터 분할: Train 70% / Validation 15% / Test 15%

**플레이리스트 K-pop 강도 분류:**

| 카테고리 | 기준 |
|----------|------|
| Pure K-pop | K-pop 비율 100% |
| K-pop Dominant | K-pop 비율 50% 이상 |
| K-pop Mixed | K-pop 비율 10% 이상 50% 미만 |
| K-pop Minimal | K-pop 비율 0% 초과 10% 미만 |
| Non K-pop | K-pop 비율 0% |

---

## 실험 결과

### 전체 성능 비교 (k=20)

| 모델 | NDCG@20 | Recall@20 |
|------|---------|-----------|
| BPR-MF | 0.0470 | 0.1293 |
| Vanilla LightGCN | 0.0482 | 0.1345 |
| **Fused LightGCN (제안)** | **0.0512** | **0.1560** |

- Fused LightGCN은 Vanilla LightGCN 대비 **NDCG@20 약 6% 향상**
- 모든 K-pop 비율 카테고리에서 Vanilla LightGCN보다 높은 성능
- 특히 K-pop Mixed 카테고리에서 NDCG 기준 **3.03%의 최대 성능 개선** 달성

---

## 하이퍼파라미터

| 파라미터 | 값 |
|----------|----|
| Embedding Dimension | 64 |
| LightGCN Layers | 3 |
| Batch Size | 2048 |
| Learning Rate | 0.001 |
| Epochs | 30 |
| Top-K (평가) | 20 |
| Fusion Alpha (`fusion_alpha`) | 0.3 |
| Group Alpha (`group_alpha`) | 0.1 |

---

## 코드 구조

`melon_code.ipynb` 파일은 아래의 단계로 구성된다.

| 셀 | 내용 |
|----|------|
| Cell 0 | 환경 설치 (`torch_geometric`) |
| Cell 1 | **Step 1**: 지식 그래프 구축 (별칭·그룹 정보 포함) |
| Cell 2 | **Step 2**: GCN 임베딩 학습 (`artist_group_embeddings_improved.pt` 출력) |
| Cell 3 | **Step 3**: 멜론-Kaggle 영어 아티스트명 매칭 |
| Cell 4 | **Step 4**: K-pop 비율 기반 층화 데이터 분할 |
| Cell 5 | **Fused LightGCN** 학습 (제안 모델) |
| Cell 6 | **Vanilla LightGCN** 학습 (비교 모델) |
| Cell 7 | **BPR-MF** 학습 (비교 모델) |
| Cell 8 | 전체 모델 성능 비교 (Precision, Recall, NDCG) |
| Cell 9 | K-pop 비율별 플레이리스트 카테고리 일관성 검증 |

---

## 환경 및 의존성

```
Python 3.x
torch
torch_geometric
numpy
pandas
scipy
tqdm
matplotlib
seaborn
```

설치:
```bash
pip install torch torch_geometric numpy pandas scipy tqdm matplotlib seaborn
```

---

## 실행 방법

1. 멜론 플레이리스트 데이터셋과 Kaggle K-pop 데이터셋을 준비한다.
2. Google Drive 경로(`/content/drive/MyDrive/melon/`)에 데이터를 배치한다.
3. `melon_code.ipynb`의 셀을 순서대로 실행한다.

---

## 기대효과

- K-pop 팬이 선호하는 아티스트의 솔로·유닛·콜라보레이션 활동을 자연스럽게 추천
- 신인 및 덜 알려진 아티스트의 노출 기회 확대
- 그룹 멤버의 신규 솔로곡 같은 **콜드 아이템 추천 문제 해결**에 기여
- K-pop 생태계의 다양성과 지속 가능성 향상

---

## 참고 문헌

1. Baek, Y.M, "Korean Soft Power: K-pop Media Consumption Through the Lens of Attraction Psychology", *Kritika Kultura*, 2016
2. Ferraro A., Kim Y., Lee S., "Melon Playlist Dataset: a public dataset for audio-based playlist generation and music tagging", *ICASSP 2021*
3. He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation", *Proceedings of the 43rd ACM SIGIR*, 639–648.
