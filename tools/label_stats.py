"""PTB-XL superdiagnostic 라벨 통계 실측 (stdlib 전용, pandas 불필요).

DESIGN.md 의 미확정 항목(실제 레코드 수, 라벨 비어서 제외되는 수, 클래스별 유병률)을
다운로드된 메타데이터 CSV 만으로 확정한다.
"""
import ast
import csv
import sys
from collections import Counter, defaultdict

ROOT = "/mnt/hdd_storage/datasets/ptb-xl/ptb-xl-1.0.3"

# 1) SCP 코드 -> superdiagnostic class 매핑 (diagnostic == 1 인 문장만)
scp2super = {}
with open(f"{ROOT}/scp_statements.csv", newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    code_col = r.fieldnames[0]
    for row in r:
        if row.get("diagnostic", "").strip() in ("1", "1.0"):
            cls = row.get("diagnostic_class", "").strip()
            if cls:
                scp2super[row[code_col].strip()] = cls
print(f"[scp_statements] diagnostic 문장 수 = {len(scp2super)}")
print(f"[scp_statements] superdiagnostic 클래스 = {sorted(set(scp2super.values()))}")

# 2) 레코드별 라벨 집계
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
n_total = 0
n_empty = 0
cls_count = Counter()
n_labels_hist = Counter()
fold_total = Counter()
fold_kept = Counter()
patients = set()
missing_fold = 0

with open(f"{ROOT}/ptbxl_database.csv", newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        n_total += 1
        patients.add(row["patient_id"])
        try:
            fold = int(float(row["strat_fold"]))
        except (ValueError, KeyError):
            fold = -1
            missing_fold += 1
        fold_total[fold] += 1

        codes = ast.literal_eval(row["scp_codes"])
        supers = {scp2super[c] for c in codes if c in scp2super}
        if not supers:
            n_empty += 1
            continue
        fold_kept[fold] += 1
        n_labels_hist[len(supers)] += 1
        for s in supers:
            cls_count[s] += 1

n_kept = n_total - n_empty
print()
print(f"[database] 전체 레코드 = {n_total}, 환자 = {len(patients)}")
print(f"[database] superdiagnostic 라벨 있음 = {n_kept}, 비어서 제외 = {n_empty} "
      f"({n_empty / n_total * 100:.2f}%)")
if missing_fold:
    print(f"[warn] strat_fold 결측 = {missing_fold}")

print()
print("클래스별 양성 수 (멀티라벨, 유지된 레코드 기준):")
for c in CLASSES:
    n = cls_count[c]
    print(f"  {c:<5} {n:>6}  ({n / n_kept * 100:5.2f}%)")
extra = set(cls_count) - set(CLASSES)
if extra:
    print(f"[warn] 예상 외 클래스: {sorted(extra)}")

print()
print("레코드당 라벨 개수 분포:")
for k in sorted(n_labels_hist):
    print(f"  {k}개: {n_labels_hist[k]:>6}")

print()
print("fold  전체    유지    (분할)")
split = {**{i: "train" for i in range(1, 9)}, 9: "val", 10: "test"}
for fold in sorted(fold_total):
    print(f"  {fold:>2}  {fold_total[fold]:>6}  {fold_kept[fold]:>6}   {split.get(fold, '?')}")

tr = sum(fold_kept[i] for i in range(1, 9))
va = fold_kept[9]
te = fold_kept[10]
print()
print(f"train(1-8) = {tr}, val(9) = {va}, test(10) = {te}, 합 = {tr + va + te}")
