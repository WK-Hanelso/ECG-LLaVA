# Confidentiality Audit

## Scope

`portfolio/` 아래의 Markdown, JSON, SVG, PNG와 생성 스크립트를 모두 검사했다. 원본 artifact의 private 절대 경로 값은 evidence package로 복사하지 않았으며, source는 repo-relative path와 JSON key로만 표기했다.

## Checks

검사 명령: `python3 portfolio/scripts/security_scan.py`

이 명령은 scanner 자신의 소스를 포함한 text file과 PNG metadata chunk를 검사한다.
실행 결과: `SECURITY_SCAN=PASS` (13 files, 3 PNG metadata chunks).

| 검사 항목 | 결과 | 조치 |
|---|---|---|
| 절대 내부 경로 및 local URI | 0건 | 없음 |
| IPv4 주소 | 0건 | 없음 |
| credential assignment / bearer material | 0건 | 없음 |
| private/internal endpoint 및 host marker | 0건 | 없음 |
| patient/record/vehicle identifier 및 messaging topic | 0건 | 없음 |
| email 형태 개인정보 | 0건 | 없음 |
| PNG text metadata | 민감 정보 0건 | 없음 |
| SVG URL review | `rg -n 'https?://' portfolio/assets/*.svg` | 표준 W3C/RDF namespace와 Matplotlib public metadata만 존재 | 포맷/도구 공개 metadata이므로 유지 |
| 이미지 해상도 | PNG header와 SVG root size 확인 | PNG 3개 모두 2400×1350; SVG 3개 모두 CSS 96 dpi 환산 width 1536px | 없음 |

## Deliberate exclusions

- `prepare_manifest.json`의 원본 데이터 저장 위치 값은 공개 문서, JSON, 이미지, 스크립트에 복사하지 않았다.
- 개별 ECG 또는 patient identifier는 사용하지 않았다.
- 저장소에 없는 hostname, IP, vehicle parameter, private API endpoint, messaging topic, secret 또는 token 값을 생성하지 않았다.
- SVG/PNG에 들어간 Matplotlib 버전과 공개 표준 namespace는 기밀 정보가 아니다.

## Result

**PASS** — 공개 산출물에서 내부 경로, 네트워크 식별자, credential, 개인정보 또는 비공개 시스템 식별자를 발견하지 않았다.
