#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SESSION = os.getenv("BAYMAX_TMUX_SESSION", "baymax")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8001

PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>大白 · 实时终端</title>
<style>
  :root{color-scheme:dark}
  *{box-sizing:border-box}
  body{margin:0;background:#14121a;color:#e8e6ef;
       font-family:"Cascadia Mono",Consolas,"Noto Sans Mono CJK SC",monospace}
  header{display:flex;align-items:center;gap:12px;padding:10px 16px;
         background:#1d1a26;border-bottom:1px solid #2e2938;font-size:13px}
  .dot{width:8px;height:8px;border-radius:50%;background:#3ddc84;
       box-shadow:0 0 8px #3ddc84;animation:pulse 1.6s infinite}
  @keyframes pulse{50%{opacity:.35}}
  .title{font-weight:600}
  .spacer{flex:1}
  .meta{color:#8d879c;font-size:12px}
  pre{margin:0;padding:16px 18px;font-size:13px;line-height:1.55;
      white-space:pre-wrap;word-break:break-all;min-height:100vh}
  .g{color:#6ee7a0}.y{color:#f4d06f}.r{color:#ff7b7b}
  .c{color:#7fd3ff}.d{color:#7d7790}
</style></head>
<body>
<header>
  <span class="dot"></span>
  <span class="title">大白 · 实时终端</span>
  <span class="spacer"></span>
  <span class="meta" id="meta">连接中…</span>
</header>
<pre id="pane"></pre>
<script>
function paint(t){
  const esc=t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  document.getElementById('pane').innerHTML=esc
    .replace(/^(▌.*)$/gm,'<span class="c">$1</span>')
    .replace(/^(我:.*)$/gm,'<span class="y">$1</span>')
    .replace(/^(大白:.*)$/gm,'<span class="g">$1</span>')
    .replace(/(❌|失败|错误|超时)/g,'<span class="r">$1</span>')
    .replace(/(✓|✅|已就绪|已连接|sent)/g,'<span class="g">$1</span>');
  window.scrollTo(0,document.body.scrollHeight);
}
let lastLen=0;
async function tick(){
  try{
    const r=await fetch('/pane?t='+Date.now());
    const j=await r.json();
    if(j.ok){
      if(j.text.length!==lastLen){ paint(j.text); lastLen=j.text.length; }
      document.getElementById('meta').textContent
        = j.alive ? ('在线 · '+j.lines+' 行 · '+j.now) : ('主程序未运行 · '+j.now);
    }
  }catch(e){
    document.getElementById('meta').textContent='连接断开，重试中…';
  }
}
setInterval(tick,800); tick();
</script>
</body></html>"""


def capture():
    try:
        r = subprocess.run(["tmux", "capture-pane", "-pt", SESSION],
                           capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return "", False, 0
        text = r.stdout.rstrip()
        lines = len([l for l in text.splitlines() if l.strip()])
        return text, True, lines
    except Exception:
        return "", False, 0


def main_alive():
    try:
        r = subprocess.run(["pgrep", "-f", "python3 -u baymax_3.3.py"],
                           capture_output=True, text=True, timeout=3)
        return bool(r.stdout.strip())
    except Exception:
        return False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        import datetime
        if self.path.startswith("/pane"):
            text, ok, lines = capture()
            body = json.dumps({
                "ok": ok,
                "text": text,
                "lines": lines,
                "alive": main_alive(),
                "now": datetime.datetime.now().strftime("%H:%M:%S"),
            }, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("实时终端监视器： http://<板子IP>:%d" % PORT)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
