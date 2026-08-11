"""PhysioNet AWS 공개 미러(physionet-open)에서 PTB-XL 1.0.3 전체 오브젝트 목록을 수집한다.
자격증명 불필요. 결과: keys_all.txt (키 목록), 표준출력에 개수/총용량 요약.
"""
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BUCKET = "https://physionet-open.s3.amazonaws.com/"
PREFIX = "ptb-xl/1.0.3/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

keys = []
total = 0
token = None
pages = 0

while True:
    params = {"list-type": "2", "prefix": PREFIX, "max-keys": "1000"}
    if token:
        params["continuation-token"] = token
    url = BUCKET + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as resp:
        root = ET.fromstring(resp.read())
    for c in root.findall(NS + "Contents"):
        keys.append(c.find(NS + "Key").text)
        total += int(c.find(NS + "Size").text)
    pages += 1
    if root.findtext(NS + "IsTruncated") == "true":
        token = root.findtext(NS + "NextContinuationToken")
    else:
        break

with open("keys_all.txt", "w") as f:
    f.write("\n".join(keys) + "\n")

n100 = sum(1 for k in keys if "/records100/" in k)
n500 = sum(1 for k in keys if "/records500/" in k)
print(f"pages={pages} objects={len(keys)} total={total/1e9:.2f} GB")
print(f"records100={n100} records500={n500} other={len(keys)-n100-n500}")
print("--- other keys ---")
for k in keys:
    if "/records100/" not in k and "/records500/" not in k:
        print(" ", k)
