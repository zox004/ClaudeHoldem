---
name: poker-ai-dev
description: HUNL 포커 AI (Deep CFR 기반) 개발 시 자동으로 활성화되어야 하는 도메인 컨벤션과 레퍼런스 패턴. Phase 4(RLCard NL Hold'em) 이후 활성화 권장. infoset 인코딩, action abstraction, reservoir buffer, 네트워크 설계 등 포커 AI 특화 코드 작성 시 참조한다.
---

# Poker AI Development Skill

HUNL Deep CFR 프로젝트 전용 개발 컨벤션. 포커 AI 구현에서 반복적으로 등장하는 패턴들을 모아둔다.

## 언제 이 skill을 사용하나

- HUNL 환경을 다루는 코드 작성 시
- Infoset 인코딩 / action abstraction 설계 시
- Deep CFR 네트워크 학습 루프 구현 시
- Reservoir buffer, self-play manager 구현 시
- 포커 도메인 용어가 등장하는 모든 작업

## 핵심 설계 원칙

### 1. Infoset 인코딩 표준

Infoset = 해당 플레이어가 **관측 가능한 모든 정보**. NLHE에서:

```python
@dataclass(frozen=True)
class InfosetKey:
    hole_cards: tuple[int, int]      # (0~51, 0~51) sorted
    board_cards: tuple[int, ...]     # flop + turn + river so far
    action_history: tuple[Action, ...]  # 모든 라운드 액션 순서
    position: int                    # 0 (SB) or 1 (BB)
    
    def to_string(self) -> str:
        """Tabular CFR용 결정론적 key"""
        return f"{self.hole_cards}|{self.board_cards}|{self.action_history}|{self.position}"
```

**Neural 인코딩** (Deep CFR용):
- **Card tensor**: 4 suits × 13 ranks 원-핫 × N 채널 (hole / flop / turn / river / all public)
- **Action tensor**: [max_actions_per_round, 4 rounds, action_features]
  - action_features = [player0_amount, player1_amount, action_type_onehot]
- 카드와 액션은 **별도 인코더**로 처리 (AlphaHoldem pseudo-Siamese 아이디어)

### 2. Action Abstraction

M1 Pro 스케일에서 권장하는 action space:

```python
class ActionSpace:
    FOLD = 0
    CALL = 1          # = check if no bet to call
    BET_HALF_POT = 2  # min-raise 처리 포함
    BET_POT = 3
    BET_2POT = 4
    ALL_IN = 5
    # 총 6개
```

- **Off-tree action mapping** (상대가 0.75 pot 베팅 시):
  ```python
  def pseudo_harmonic_mapping(actual_bet: float, a: float, b: float) -> int:
      """Ganzfried & Sandholm 2013. a < actual < b 사이 보간."""
      prob_a = ((b - actual_bet) * (1 + a)) / ((b - a) * (1 + actual_bet))
      return SIZE_A if random.random() < prob_a else SIZE_B
  ```

### 3. Card Abstraction (HUNL에서 선택적)

Leduc까지는 필요 없음. HUNL에서 성능 필요 시:

```python
# E[HS²] 버킷팅: 라운드당 bucket 개수
BUCKETS_PER_ROUND = {
    "preflop": 169,   # starting hand (isomorphism)
    "flop": 200,
    "turn": 200,
    "river": 200,
}
```

Precompute해서 disk에 저장 (~수백 MB). `eval7` 라이브러리 활용.

### 4. Reservoir Buffer 올바른 구현

**잘못된 구현 주의**: 단순 FIFO나 ring buffer는 Deep CFR 안 됨. **실제 reservoir sampling** 필요.

```python
class ReservoirBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.entries = []
        self.num_seen = 0  # 전체 본 샘플 수
    
    def add(self, entry):
        self.num_seen += 1
        if len(self.entries) < self.capacity:
            self.entries.append(entry)
        else:
            # 핵심: 확률 capacity/num_seen으로 교체
            idx = random.randint(0, self.num_seen - 1)
            if idx < self.capacity:
                self.entries[idx] = entry
    
    def sample(self, batch_size):
        return random.sample(self.entries, min(batch_size, len(self.entries)))
```

### 5. 네트워크 아키텍처 (Deep CFR)

**M1 Pro에서 돌아가는 현실적 크기**:

```python
class AdvantageNet(nn.Module):
    def __init__(self, card_channels=6, action_channels=24, num_actions=6):
        super().__init__()
        # Card branch
        self.card_conv = nn.Sequential(
            nn.Conv2d(card_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        # Action branch (AlphaHoldem 스타일)
        self.action_conv = nn.Sequential(
            nn.Conv2d(action_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        # Fusion
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 13 + 64 * 4 * 9, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_actions),  # advantage output
        )
    
    def forward(self, cards, actions):
        c = self.card_conv(cards)
        a = self.action_conv(actions)
        return self.fc(torch.cat([c, a], dim=-1))
```

**파라미터 수**: 약 2~3M. M1 Pro MPS에서 배치 256 학습 가능.

### 6. Self-Play Pool (K-best)

```python
class KBestPool:
    """체크포인트 pool 중 상위 K개 ELO 유지"""
    def __init__(self, k=5):
        self.k = k
        self.pool = []  # [(checkpoint_path, elo), ...]
    
    def add(self, checkpoint_path, elo):
        self.pool.append((checkpoint_path, elo))
        self.pool.sort(key=lambda x: -x[1])  # 내림차순
        self.pool = self.pool[:self.k]
    
    def sample_opponent(self) -> str:
        """K개 중 랜덤 선택 (ELO 가중 가능)"""
        return random.choice(self.pool)[0]
```

### 7. Evaluation 단위 일관성

**항상 mbb/hand**:

```python
def compute_mbb_per_hand(
    total_winnings_chips: float,
    num_hands: int,
    big_blind_chips: float,
) -> float:
    bb_per_hand = total_winnings_chips / (num_hands * big_blind_chips)
    return bb_per_hand * 1000  # convert to milli-bb
```

### 8. Determinism 확보

```python
def set_all_seeds(seed: int):
    import random, numpy as np, torch, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed) if torch.backends.mps.is_available() else None
    os.environ["PYTHONHASHSEED"] = str(seed)
```

## 흔한 함정 (Common Pitfalls)

1. **Dict 순회 순서 의존**: Python 3.7+에서도 infoset key를 dict 순회로 만들면 플랫폼 의존성 생김. 명시적으로 sort.
2. **Log(0) 에러**: `log(prob)` 계산 시 `torch.clamp(prob, min=1e-10)` 필수
3. **Legal mask 안 씀**: `softmax` 전에 illegal action에 `-inf` 넣기
4. **Terminal utility 시점 혼란**: fold vs showdown의 pot 계산 분리
5. **MPS float64 이슈**: 가능하면 float32로 통일
6. **Reservoir buffer에 iteration 태깅 안 함**: Deep CFR은 iter 정보가 target에 들어감

## 체크리스트: 새 모듈 작성 시

- [ ] Dataclass는 `frozen=True` (infoset key 등 해시 가능해야 함)
- [ ] Random seed가 함수 인자 또는 클래스 멤버로 명시
- [ ] Legal action mask를 네트워크 출력에 적용
- [ ] 단위 (chips vs bb vs mbb) 명확히 주석
- [ ] Checkpoint에 buffer state도 포함 (resume 가능하게)
- [ ] W&B 로깅 키 컨벤션 따름 (`train/*`, `eval/*`, `debug/*`)

## 참조 저장소 (구현 패턴 확인용)

- **EricSteinberger/Deep-CFR**: Deep CFR 논문 저자 구현. PyTorch.
- **EricSteinberger/PokerRL**: 분산 self-play 프레임워크
- **google-deepmind/open_spiel**: CFR 계열 참조 구현 (C++/Python)
- **datamllab/rlcard**: NLHE 환경, NFSP 참조

## 디버깅 우선 순위 (문제 생겼을 때)

1. Legal mask 적용 확인
2. Infoset key 결정론 확인 (같은 히스토리 → 같은 key)
3. Regret / advantage 부호 확인
4. Reservoir buffer가 실제로 reservoir인지
5. Network gradient 흐름 확인 (`requires_grad`)
6. Loss 스케일이 0.01~100 범위인지 (너무 작거나 크면 의심)

---

**이 skill이 활성화되면 위 원칙을 자동으로 적용하여 코드 작성 및 리뷰를 수행한다.**
