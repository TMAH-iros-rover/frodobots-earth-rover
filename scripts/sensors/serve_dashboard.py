"""
serve_dashboard.py
scripts/sensors/index.html 을 로컬 서버로 띄워서 브라우저에서 라이브 로버 대시보드를
보게 해준다. 정적 파일 서빙 + scripts/perception/live_bridge.py 의 실시간 traversability
/path 추론 결과(GET /traversability)를 **같은 포트 하나**에서 같이 서빙한다 (전에는
perception 브릿지를 9222 포트에 따로 띄웠었는데, 대시보드 포트 하나로 다 보고 싶다는
요청에 따라 합침).

전제:
  earth-rovers-sdk 컨테이너(docker compose)가 이미 떠 있고, http://localhost:8000
  에서 /data, /v2/front 를 서빙하고 있어야 함.
  torch/transformers(Sam2Model 포함 버전)가 설치돼 있어야 함 (컨테이너 안에는 이미
  있음; host에는 없을 수 있으니 이 스크립트는 컨테이너 안에서 실행해야 한다).

사용 (컨테이너 안에서):
  python3 scripts/sensors/serve_dashboard.py
      # 기본: --estimator sam_tp, 체크포인트는 scripts/perception/models/best_sam_tp.pt
  python3 scripts/sensors/serve_dashboard.py --estimator heuristic   # 가볍게 테스트할 때
  -> http://localhost:5050 접속
"""
import argparse
import functools
import json
import os
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PERCEPTION_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "perception")
sys.path.insert(0, PERCEPTION_DIR)
import live_bridge  # noqa: E402  (perception 폴더를 sys.path에 넣은 뒤에 import)


class DashboardHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass  # 정적 파일 요청 스팸 방지 (에러는 그대로 stderr에 남음)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/traversability"):
            self._send_json(live_bridge.latest.get())
            return
        if self.path.startswith("/health"):
            self._send_json({"status": "ok"})
            return
        super().do_GET()


def main():
    parser = argparse.ArgumentParser(parents=[live_bridge.build_arg_parser()])
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    print(f"[perception] estimator={args.estimator}  sdk={args.sdk_base_url}  "
          f"checkpoint={args.sam_tp_checkpoint if args.estimator == 'sam_tp' else '(해당없음)'}")
    estimator = live_bridge.build_estimator(args)
    threading.Thread(target=live_bridge.infer_loop, args=(args, estimator), daemon=True).start()

    handler = functools.partial(DashboardHandler, directory=SCRIPT_DIR)
    with ThreadingHTTPServer(("0.0.0.0", args.port), handler) as httpd:
        print(f"[dashboard] http://localhost:{args.port} 에서 서빙 중 (Ctrl+C로 종료)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[dashboard] 종료")


if __name__ == "__main__":
    main()
