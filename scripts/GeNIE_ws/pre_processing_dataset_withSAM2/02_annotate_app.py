"""
scripts/GeNIE_ws/pre_processing_dataset_withSAM2/02_annotate_app.py
GeNIE 논문(Sec III-B) 반자동 라벨링 파이프라인의 2단계: "사람이 SAM2 후보 중
주행 가능한 영역을 선택/수정"을 로컬 웹앱으로 구현한다.

    (01번이 만든) SAM2 후보 마스크들 -> [이 웹앱에서 사람이 클릭으로 선택] -> 최종 traversability mask

사용법 (실행 후 브라우저에서 http://<이 머신의 IP 또는 localhost>:5050 접속):
  python3 scripts/GeNIE_ws/pre_processing_dataset_withSAM2/02_annotate_app.py
  (원격 서버에서 돌리는 중이면 SSH 포트포워딩: ssh -L 5050:localhost:5050 <host>)

화면 구성:
  - 프레임 이미지 위에, 01번이 만든 후보 영역들의 경계선을 얇은 흰 선으로 항상
    표시해둔다 (전체적으로 어떤 영역들이 후보인지 한눈에 보이게).
  - 오른쪽 체크박스 목록에서 "주행 가능한 영역"에 해당하는 후보를 체크하면, 그
    영역이 이미지 위에 초록색으로 실시간 하이라이트된다 (여러 개 선택 가능 ->
    선택된 영역들의 합집합이 최종 마스크가 됨).
  - "이 프레임엔 주행 가능 영역 없음" 체크박스: 빈(전부 0) 마스크로 저장하고 싶을 때.
  - 저장하면 다음 미완료(annotated=0) 프레임으로 자동 이동한다.

저장 결과:
  - data/labels/<sample_id>.png : 최종 traversability mask (0=불가, 255=가능), 1채널
  - data/manifest.csv 의 annotated 컬럼이 1로 갱신됨

이 단계에서 사람이 직접 픽셀을 칠하는 프리핸드 보정까지는 지원하지 않는다(후보
선택/합치기까지만) — 01번의 격자 프롬프트가 촘촘해서(기본 7x6=42점) 대부분의
장면에서 "바닥 전체"에 해당하는 후보가 이미 하나로 잘 뽑히는 걸 확인했다. 만약
후보들 중 어느 것도 원하는 경계와 안 맞으면, 일단 최선의 근사치로 선택해 저장하고
넘어가거나 이 프레임을 건너뛰면 된다(체크 없이 저장 시 빈 마스크로 저장됨).
"""
import os
import csv
import io

import cv2
import numpy as np
from flask import Flask, request, redirect, url_for, send_file, Response

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "data")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
PROPOSALS_DIR = os.path.join(DATA_DIR, "proposals")
LABELS_DIR = os.path.join(DATA_DIR, "labels")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")

os.makedirs(LABELS_DIR, exist_ok=True)

app = Flask(__name__)


# --------------------------------------------------------------------------
# manifest.csv 입출력
# --------------------------------------------------------------------------

def read_manifest():
    rows = []
    if os.path.isfile(MANIFEST_PATH):
        with open(MANIFEST_PATH, newline="") as f:
            for row in csv.DictReader(f):
                row["annotated"] = int(row["annotated"])
                rows.append(row)
    return rows


def write_manifest(rows):
    if not rows:
        return
    header = list(rows[0].keys())
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def load_masks(sample_id):
    """proposals/<sample_id>.npz -> (masks: (K,H,W) bool ndarray, scores: (K,))."""
    d = np.load(os.path.join(PROPOSALS_DIR, f"{sample_id}.npz"))
    k, h, w = int(d["k"]), int(d["h"]), int(d["w"])
    if k == 0:
        return np.zeros((0, h, w), dtype=bool), np.zeros((0,), dtype=np.float32)
    bits = np.unpackbits(d["packed"])[: k * h * w]
    masks = bits.reshape(k, h, w).astype(bool)
    return masks, d["scores"]


# --------------------------------------------------------------------------
# 이미지 렌더링 (경계선 오버레이 / 개별 마스크 하이라이트 PNG)
# --------------------------------------------------------------------------

def _encode_png(img_bgr_or_bgra):
    ok, buf = cv2.imencode(".png", img_bgr_or_bgra)
    return io.BytesIO(buf.tobytes())


def render_outline(sample_id):
    """모든 후보 영역의 경계선(흰 선) + 번호(체크박스 목록과 대조할 수 있게)를 그린
    기본 배경 이미지."""
    frame = cv2.imread(os.path.join(FRAMES_DIR, f"{sample_id}.jpg"))
    masks, _ = load_masks(sample_id)
    out = frame.copy()
    for i, m in enumerate(masks):
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
        ys, xs = np.where(m)
        if len(xs):
            cy, cx = int(ys.mean()), int(xs.mean())
            cv2.putText(out, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_indexmap(sample_id):
    """픽셀마다 '어느 마스크 번호에 속하는지'를 담은 1채널 이미지(0=배경, i+1=마스크
    i). 프론트엔드에서 이미지를 클릭했을 때 어떤 마스크를 클릭했는지 알아내는 용도
    (자바스크립트 canvas로 이 이미지의 픽셀값을 읽는다). 큰 마스크를 먼저 칠하고
    작은 마스크를 나중에 덮어써서, 영역이 겹칠 때는 더 구체적인(작은) 마스크가
    클릭 우선순위를 갖게 한다."""
    masks, _ = load_masks(sample_id)
    if len(masks) == 0:
        return np.zeros((2, 2), dtype=np.uint8)
    h, w = masks.shape[1:]
    idxmap = np.zeros((h, w), dtype=np.uint8)
    order = sorted(range(len(masks)), key=lambda i: -masks[i].sum())  # 큰 것부터 -> 작은 게 마지막에 위에 그려짐
    for i in order:
        idxmap[masks[i]] = i + 1
    return idxmap


def render_mask_layer(sample_id, idx, color=(80, 220, 80)):
    """마스크 하나를 RGBA 투명 PNG로: 마스크 영역만 색+alpha, 나머지는 완전 투명.
    체크박스 토글 시 이 이미지를 <img>로 겹쳐 보이거나/숨기는 식으로 프론트에서 씀."""
    masks, _ = load_masks(sample_id)
    h, w = masks.shape[1:] if len(masks) else (2, 2)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if 0 <= idx < len(masks):
        m = masks[idx]
        rgba[m, 0] = color[0]
        rgba[m, 1] = color[1]
        rgba[m, 2] = color[2]
        rgba[m, 3] = 140
    return rgba


# --------------------------------------------------------------------------
# 라우트
# --------------------------------------------------------------------------

@app.route("/")
def index():
    rows = read_manifest()
    if not rows:
        return "manifest.csv가 비어있음 -> 먼저 01_generate_mask_proposals.py 를 실행해줘."
    pending = [r for r in rows if not r["annotated"]]
    target = pending[0] if pending else rows[0]
    return redirect(url_for("annotate", sample_id=target["sample_id"]))


@app.route("/annotate/<sample_id>")
def annotate(sample_id):
    rows = read_manifest()
    row = next((r for r in rows if r["sample_id"] == sample_id), None)
    if row is None:
        return f"sample_id={sample_id} 를 manifest에서 못 찾음", 404

    masks, scores = load_masks(sample_id)
    done = sum(r["annotated"] for r in rows)
    total = len(rows)

    checkboxes = "".join(
        f'<label class="chk"><input type="checkbox" id="cb{i}" name="mask_id" value="{i}" '
        f'onchange="setMaskSelected({i}, this.checked)"> #{i} (score {s:.2f})</label><br>'
        for i, s in enumerate(scores)
    )

    html = f"""
    <html><head><title>traversability 라벨링: {sample_id}</title>
    <style>
      body {{ font-family: sans-serif; background:#111; color:#eee; }}
      #stage {{ position: relative; display:inline-block; cursor: pointer; }}
      /* 기준(outline) 이미지는 일반 흐름에 둬서 #stage가 실제 이미지 크기만큼
         커지게 하고, 마스크 오버레이 레이어들만 그 위에 절대좌표로 겹친다.
         (예전엔 outline까지 position:absolute라 #stage 박스가 0x0으로 찌그러져서
         클릭 좌표 계산/히트테스트가 안 먹히는 버그가 있었음) */
      #stage img {{ display:block; pointer-events:none; }}
      #stage .layer {{ position:absolute; top:0; left:0; display:none; }}
      .panel {{ display:inline-block; vertical-align:top; margin-left:24px; }}
      .chk {{ display:block; padding:2px 0; }}
      button {{ font-size:16px; padding:8px 16px; margin-top:12px; cursor:pointer; }}
      #hint {{ color:#8f8; font-size:13px; }}
    </style></head>
    <body>
      <h3>[{done}/{total} 완료] {sample_id} (ride {row['ride']}, frame {row['frame_idx']}, 후보 {len(scores)}개)</h3>
      <p id="hint">이미지 위에서 원하는 영역을 <b>직접 클릭</b>해도 되고, 오른쪽 체크박스로 골라도 됩니다
      (숫자는 체크박스 목록의 #번호와 대응). 다시 클릭하면 선택 해제됩니다.</p>
      <div id="stage">
        <img src="/media/{sample_id}/outline.png">
        {"".join(f'<img id="layer{i}" class="layer" src="/media/{sample_id}/mask/{i}.png">' for i in range(len(scores)))}
      </div>
      <div class="panel">
        <b>주행 가능한 영역 선택 (여러 개 가능):</b><br><br>
        {checkboxes}
        <br>
        <label><input type="checkbox" id="none_traversable"> 이 프레임엔 주행 가능 영역 없음</label>
        <br>
        <button onclick="save()">저장 &amp; 다음</button>
        <button onclick="skip()">건너뛰기(저장 안 함)</button>
      </div>
      <script>
        function setMaskSelected(i, on) {{
          document.getElementById("layer" + i).style.display = on ? "block" : "none";
          document.getElementById("cb" + i).checked = on;
        }}

        // 클릭한 픽셀이 어느 마스크에 속하는지 알아내기 위해, 서버가 만든 indexmap.png
        // (픽셀값 = 마스크번호+1, 0=배경)를 캔버스에 그려서 픽셀 데이터를 읽어둔다.
        let idxData = null, idxW = 0, idxH = 0;
        (function loadIndexMap() {{
          const img = new Image();
          img.onload = function() {{
            const c = document.createElement("canvas");
            c.width = img.naturalWidth; c.height = img.naturalHeight;
            const ctx = c.getContext("2d");
            ctx.drawImage(img, 0, 0);
            idxData = ctx.getImageData(0, 0, c.width, c.height).data;
            idxW = c.width; idxH = c.height;
          }};
          img.src = "/media/{sample_id}/indexmap.png";
        }})();

        document.getElementById("stage").addEventListener("click", function(ev) {{
          if (!idxData) {{ alert("아직 인덱스맵 로딩 중입니다. 잠시 후 다시 클릭해주세요."); return; }}
          const rect = this.getBoundingClientRect();
          const x = Math.floor(ev.clientX - rect.left);
          const y = Math.floor(ev.clientY - rect.top);
          if (x < 0 || y < 0 || x >= idxW || y >= idxH) return;
          const val = idxData[(y * idxW + x) * 4];  // 그레이스케일이라 R=G=B
          if (val > 0) {{
            const i = val - 1;
            const cb = document.getElementById("cb" + i);
            setMaskSelected(i, !cb.checked);
          }}
        }});

        function save() {{
          const ids = Array.from(document.querySelectorAll('input[name="mask_id"]:checked')).map(el => el.value);
          const none = document.getElementById("none_traversable").checked;
          fetch("/save/{sample_id}", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify({{mask_ids: none ? [] : ids}})
          }}).then(r => r.json()).then(d => {{ window.location.href = d.next; }});
        }}
        function skip() {{ window.location.href = "/next/{sample_id}"; }}
      </script>
    </body></html>
    """
    return html


@app.route("/media/<sample_id>/outline.png")
def media_outline(sample_id):
    img = render_outline(sample_id)
    return send_file(_encode_png(img), mimetype="image/png")


@app.route("/media/<sample_id>/indexmap.png")
def media_indexmap(sample_id):
    img = render_indexmap(sample_id)
    return send_file(_encode_png(img), mimetype="image/png")


@app.route("/media/<sample_id>/mask/<int:idx>.png")
def media_mask(sample_id, idx):
    img = render_mask_layer(sample_id, idx)
    return send_file(_encode_png(img), mimetype="image/png")


@app.route("/save/<sample_id>", methods=["POST"])
def save(sample_id):
    payload = request.get_json(force=True)
    mask_ids = [int(i) for i in payload.get("mask_ids", [])]

    masks, _ = load_masks(sample_id)
    if len(masks) == 0:
        h, w = 2, 2
    else:
        h, w = masks.shape[1:]
    final = np.zeros((h, w), dtype=bool)
    for i in mask_ids:
        if 0 <= i < len(masks):
            final |= masks[i]

    cv2.imwrite(os.path.join(LABELS_DIR, f"{sample_id}.png"), (final.astype(np.uint8) * 255))

    rows = read_manifest()
    for r in rows:
        if r["sample_id"] == sample_id:
            r["annotated"] = 1
    write_manifest(rows)

    pending = [r for r in rows if not r["annotated"]]
    next_id = pending[0]["sample_id"] if pending else None
    next_url = url_for("annotate", sample_id=next_id) if next_id else url_for("done")
    return {"ok": True, "next": next_url}


@app.route("/next/<sample_id>")
def next_sample(sample_id):
    rows = read_manifest()
    ids = [r["sample_id"] for r in rows]
    i = ids.index(sample_id) if sample_id in ids else -1
    pending = [r for r in rows if not r["annotated"]]
    if pending:
        return redirect(url_for("annotate", sample_id=pending[0]["sample_id"]))
    nxt = ids[(i + 1) % len(ids)] if ids else None
    return redirect(url_for("annotate", sample_id=nxt)) if nxt else redirect(url_for("done"))


@app.route("/done")
def done():
    rows = read_manifest()
    n = sum(r["annotated"] for r in rows)
    return f"<h2>모든 프레임({n}/{len(rows)}) 라벨링 완료!</h2><p>결과: {LABELS_DIR}</p>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
