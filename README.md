# 🧊 Image → 3D Agent

이미지 한 장을 **3D 모델**로 변환하는 로컬 도구입니다.
단일 변환은 물론, **수십~수백 장을 한 번에 처리하면서 "깨진 3D 모델"을 AI가 스스로 판별하고
자동으로 재생성하는 에이전트**까지 포함합니다.

## 🔀 두 가지 3D 엔진

| 엔진 | 출력 | 런처 | 포트 | 비고 |
|---|---|---|---|---|
| 🧱 **Hunyuan3D 2.1** (기본) | **메시 + (텍스처)** `.glb` | `run_mesh_app.bat` / `run_mesh_agent.bat` | 7861 | shape 동작 ✅ / 텍스처는 CUDA 빌드 필요 |
| 🧊 **TripoSplat** (스플랫) | 가우시안 스플랫 `.ply/.splat` | `run_app.bat` / `run_agent.bat` | 7860 | 전체 동작 ✅ |

두 엔진 모두 동일한 **배치 에이전트 + 자동 QA + 자가복구 + 결과 브라우저** 위에 올라가 있고,
각자 격리된 가상환경(`venv_hy` / `venv`)을 씁니다.

> 엔진: [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) ·
> [VAST-AI-Research/TripoSplat](https://github.com/VAST-AI-Research/TripoSplat)
> 배치 에이전트 / 메시·스플랫 QA / 자가 복구 파이프라인: **AI FUTURE STREAMER 박성우 (Park Seong-Woo)**

---

## 🧱 Hunyuan3D 2.1 메시 파이프라인 (기본)

이미지 → **고품질 3D 메시(.glb)**. 출력이 가우시안 스플랫이 아니라 **삼각형 메시**라
QA·뷰어가 메시 전용으로 새로 구현돼 있습니다.

```bat
run_mesh_app.bat
```
→ http://127.0.0.1:7861 (탭: Single image · Batch Agent · Results Browser)

배치(폴더 → 메시 대량 생성, 자동 QA + 자가복구 + 이어하기):
```bat
run_mesh_agent.bat  C:\내이미지폴더
run_mesh_agent.bat  C:\내이미지폴더  C:\출력폴더  --steps 30 --octree-resolution 384
```
출력: `mesh_outputs\success\<이름>\model.glb` (+ preprocessed/preview/info.json), `failed\`, `manifest.json`.

**메시 QA**: 빈 메시·degenerate(면수 부족)·비유한값·collapse·flat(축비율≈0)·과도 분절을
기하 검사 + 표면샘플 4방향 렌더로 판정하고, 깨지면 시드/스텝/guidance를 바꿔 재생성합니다.

### ⚠️ 텍스처(PBR)는 현재 보류
Hunyuan 텍스처 파이프라인은 **custom CUDA 래스터라이저 빌드**가 필요합니다
(이 머신엔 CUDA Toolkit·MSVC 미설치). 그래서 현재는 **shape(형상)** 만 생성합니다.
텍스처를 켜려면 **CUDA Toolkit 12.4 + Visual Studio Build Tools**를 설치한 뒤
`hy3dpaint/custom_rasterizer`와 `DifferentiableRenderer`를 빌드해야 합니다.

설치 메모(이 폴더는 이미 세팅됨, 가중치는 `~/.cache/hy3dgen`에 자동 다운로드):
```bash
py -3.12 -m venv venv_hy
venv_hy/Scripts/python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
venv_hy/Scripts/python.exe -m pip install "numpy==1.26.4" transformers==4.46.0 diffusers==0.30.0 accelerate omegaconf einops safetensors "huggingface-hub==0.30.2" trimesh scikit-image scipy opencv-python rembg onnxruntime pymeshlab timm torchdiffeq "gradio==5.33.0"
```

---

## 🧊 TripoSplat 가우시안 스플랫 파이프라인

---

## ✨ 주요 기능

| 기능 | 설명 |
|---|---|
| 🖼️ **단일 이미지 → 3D** | 사진 1장을 고품질 3D 스플랫으로 변환 |
| 🤖 **배치 에이전트** | 폴더 안의 이미지(예: 100장)를 순차 처리 |
| 🩺 **AI 자동 파손 판별** | 빈 결과·붕괴·폭발·노이즈·형상 불일치 자동 감지 |
| ♻️ **자가 복구 재생성** | 깨지면 시드/파라미터를 바꿔 **최대 4회** 재시도 |
| 📁 **실패 케이스 관리** | 4회 실패 시 기록 후 다음 이미지로 진행 (멈추지 않음) |
| ⏯️ **이어하기(Resume)** | 중단돼도 끝난 이미지는 건너뛰고 이어서 처리 |
| 👁️ **미리보기 렌더** | 원본 + 4방향 렌더 이미지를 자동 저장해 눈으로 확인 |

---

## 💻 요구 사항

- **NVIDIA GPU** (CUDA) — 권장 VRAM 12GB 이상 (개발/테스트: RTX 4090 24GB)
- **Windows 10/11** (다른 OS도 TripoSplat 자체는 동작)
- **Python 3.10~3.12**
- 디스크 여유 ~10GB (모델 가중치 약 3.6GB + PyTorch)

GPU가 정상 인식되는지 확인:
```
nvidia-smi
```

---

## 🛠️ 설치

```bash
# 1) 저장소 클론
git clone https://github.com/VAST-AI-Research/TripoSplat.git
cd TripoSplat

# 2) 가상환경
python -m venv ../venv
../venv/Scripts/python.exe -m pip install --upgrade pip

# 3) PyTorch (CUDA 12.4 빌드 예시 — 본인 환경에 맞게)
../venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4) 의존성
../venv/Scripts/python.exe -m pip install numpy safetensors pillow tqdm huggingface_hub gradio anthropic

# 5) 모델 가중치 다운로드 (~3.6GB)
../venv/Scripts/hf.exe download VAST-AI/TripoSplat --local-dir ckpts/
```

> 이 폴더(`imageto3d`)에는 위 과정이 이미 세팅되어 있습니다. 바로 아래 `.bat` 파일만 쓰면 됩니다.

---

## 🚀 사용법

### 1. 단일 이미지 → 3D
```bat
run_image.bat  C:\경로\내이미지.png
```
- 결과: `TripoSplat\output.ply`, `output.splat`
- 가우시안 수 조절(가벼움): `run_image.bat 내이미지.png 131072`

### 2. 🤖 배치 에이전트 (메인 기능)
폴더 안 모든 이미지를 순차 처리하고, 깨진 모델은 자동 재생성합니다.
```bat
run_agent.bat  C:\내이미지폴더
run_agent.bat  C:\내이미지폴더  C:\출력폴더  --steps 40
```

### 3. 🖥️ 통합 웹 앱 (권장 · 배치까지 GUI로)
```bat
run_app.bat
```
→ 브라우저에서 http://127.0.0.1:7860

탭 3개로 구성됩니다 — CLI 없이 플래그십 기능을 그대로 씁니다:
| 탭 | 기능 |
|---|---|
| **Single image** | 이미지 1장 → 3D + QA 판정/미리보기/뷰어/다운로드 |
| **Batch Agent** | 폴더 경로 입력(또는 다중 업로드) → 자동 QA·자가복구 배치 실행, **실시간 진행 표 + 성공/실패 갤러리** (이어하기 자동) |
| **Results Browser** | 기존 출력 폴더 로드 → 성공/실패·QA 사유·다운로드 검토. **성공 항목 클릭 시 3D 뷰어 인라인 표시**, **실패 케이스만 골라 재시도**(자동 복구) 버튼. (로드/뷰어는 GPU 불필요) |

> 단일 변환만 쓰는 구버전 UI: `run_webui.bat` (그대로 유지)

---

## 📂 에이전트 출력 구조

```
출력폴더\
├─ success\<이름>\
│    ├─ model.ply          # 3D 가우시안 스플랫
│    ├─ model.splat
│    ├─ preprocessed.webp  # 배경 제거된 입력
│    ├─ preview.webp       # 원본 + 4방향 렌더 (품질 확인용)
│    └─ info.json          # 시도 내역 / QA 지표
├─ failed\<이름>\          # 4회 모두 실패한 케이스
│    ├─ last_preview.webp
│    └─ info.json
├─ manifest.json           # 전체 요약 (이어하기 기준)
└─ agent.log
```

생성된 `.ply` / `.splat` 은 아래에서 바로 볼 수 있습니다:
- https://superspl.at/editor
- https://sparkjs.dev

---

## 🩺 AI 파손 판별(QA) 작동 원리

3단계로 "이 3D 모델이 깨졌는가?"를 판정합니다.

1. **기하 검사** (필수, 의존성 없음)
   NaN/Inf, 빈 결과(가시 가우시안 비율), 한 점으로 붕괴, 폭발(scale 과다)을 정상 모델 실측값
   (가시율 ~0.95, 공간범위 ~1.0, scale 중앙값 ~0.0013) 기준으로 검사.
2. **자체 4방향 렌더 검사**
   순수 PyTorch z-buffer 렌더로 커버리지·구조 이상을 감지하고 미리보기를 저장.
3. **Claude 비전 판정** (선택)
   원본 사진과 4방향 렌더를 비교해 "형상이 맞는지" 판단.
   ```bat
   set ANTHROPIC_API_KEY=sk-ant-...
   run_agent.bat C:\내이미지폴더
   ```
   키가 없으면 1·2단계만으로 정상 동작합니다.

---

## ⚙️ 에이전트 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--max-attempts` | `4` | 이미지당 최대 재시도 횟수 |
| `--steps` | `30` | 샘플러 스텝 (↑ 품질, ↑ 시간) |
| `--guidance` | `3.5` | CFG 강도 |
| `--num-gaussians` | `262144` | 가우시안 수 (최대) |
| `--qa-vision` | `auto` | `auto`/`on`/`off` (API 키 있으면 자동 사용) |
| `--limit` | `0` | 앞에서 N장만 처리 (테스트용) |
| `--resume` | (런처 기본 ON) | 끝난 이미지 건너뛰기 |

성능 참고: RTX 4090에서 약 **9초/장**(steps 16) ~ **12–18초/장**(steps 30).
빠르게/가볍게: `--steps 20 --num-gaussians 131072`.

---

## 🧩 트러블슈팅

- **`cuda available: False`** → GPU 드라이버 / PyTorch CUDA 빌드 확인 (`nvidia-smi`).
- **`nvidia-smi` 없음** → `C:\Windows\System32\nvidia-smi.exe` 직접 실행.
- **메모리 부족(OOM)** → `--num-gaussians 131072` 또는 `65536`으로 낮추기.
- **비전 QA가 동작 안 함** → `ANTHROPIC_API_KEY` 미설정이면 자동 생략(정상).

---

## 👤 개발자

### AI FUTURE STREAMER — 박성우 (Park Seong-Woo)
**AI Futurist · Future AI Engineer.**
인공지능 · 인간 창의성 · 피지컬 인텔리전스 · 자율 시스템이 융합하는 미래를 만드는 엔지니어/창업가.

- 🏢 **AIMZ Media** — 공동창업자 & VP · GenAI 콘텐츠 엔지니어링/자동화 플랫폼
  (오리지널 IP 애니메이션 제작, 3DGS·NeRF·생성형 AI 기반 멀티이미지 2D/3D 스티칭·공간 콘텐츠, 이미지 복원/업스케일, 노드 기반 AI 워크플로우)
- 🤖 **DPAX.AI** — 어드바이저 · 피지컬 AI & 로보틱스
  (외골격 하드웨어, 인간 모션 데이터, 로봇 통합·제어, XR 데이터 수집, Physical AI 학습 인프라)
- 📊 **DeepAgent** — 창업자/어드바이저 · 자율 인텔리전스 & 포사이트 플랫폼
  (미래예측 리서치, 멀티에이전트, 금융·투자 리서치, 전략 시나리오 모델링)

> **Vision** — 인간 지능 · 인공지능 · 자율 시스템의 융합으로, 물리·디지털·사회 전반에 걸쳐
> 지식을 확장하고 혁신을 가속하는 인간–AI 협업의 미래를 만든다. 🚀

---

## 📜 라이선스 / 크레딧

- 3D 생성 엔진: **TripoSplat** © VAST-AI-Research — [MIT License](https://github.com/VAST-AI-Research/TripoSplat/blob/main/LICENSE)
- 배치 에이전트 · 자동 QA · 자가 복구 파이프라인: © 박성우 (AI FUTURE STREAMER)

논문: *Generative 3D Gaussians with Learned Density Control* (arXiv:2605.16355)
