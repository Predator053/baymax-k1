#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""大白 3.3 网页面板（没有触摸屏时的替代方案）。

    python3 baymax_panel.py [端口]

电脑或手机浏览器打开 http://<K1的IP>:8000 就能看到：
  - 大白状态与最近一轮对话（含分项时延）
  - 性能看板：唤醒→出声、端到端平均、缓存命中率、累计轮次
  - 健康档案 / 健康数据 / 用药提醒 / 硬件状态
  - 演示按钮：模拟跌倒 / 测心率血氧 / 让它说一句 / 停止播报

设计取向：暖白、圆角、柔和阴影，呼应「大白」本身的气质——安静、可靠、不刺眼。
它只读写 ~/.baymax/ 下的文件，和语音主程序解耦：面板崩了不影响说话。
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baymax_service as svc
try:
    import baymax_health as health
except Exception:
    health = None

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
_STACK_CACHE = {"ts": 0.0, "data": {}}

PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>大白 Baymax · 陪护面板</title>
<style>
:root{
  --bg1:#FFFBF7; --bg2:#F6EDE4;
  --card:#FFFFFF; --ink:#3B3835; --muted:#9C948C; --line:#F0E6DC;
  --accent:#E8836A; --accent-soft:#FBE3D8; --ok:#5FAE8B; --warn:#E0A458; --bad:#D9736B;
  --shadow:0 10px 30px rgba(150,120,100,.10);
}
*{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0;color:var(--ink);
  background:
    radial-gradient(1200px 600px at 15% -10%, #FFF6EE 0%, rgba(255,246,238,0) 60%),
    radial-gradient(900px 500px at 100% 0%, #FDF0E6 0%, rgba(253,240,230,0) 55%),
    linear-gradient(180deg,var(--bg1),var(--bg2));
  font-family:-apple-system,BlinkMacSystemFont,"Noto Sans SC","PingFang SC","Microsoft YaHei",sans-serif;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1240px;margin:0 auto;padding:22px 22px 40px}

/* ---------- 顶部 ---------- */
header{display:flex;align-items:center;gap:16px;padding:6px 4px 22px}
.face{width:56px;height:56px;border-radius:50%;background:#fff;border:1px solid var(--line);
  box-shadow:0 8px 20px rgba(180,150,130,.18);position:relative;flex:0 0 auto;
  animation:breathe 4.5s ease-in-out infinite}
.face .eye{position:absolute;top:43%;width:6px;height:6px;border-radius:50%;background:#2E2B29}
.face .eye.l{left:31%} .face .eye.r{right:31%}
.face .mouth{position:absolute;top:60%;left:50%;transform:translateX(-50%);width:14px;height:1.6px;
  background:#2E2B29;border-radius:2px}
@keyframes breathe{0%,100%{transform:scale(1)}50%{transform:scale(1.045)}}
h1{font-size:20px;margin:0;font-weight:650;letter-spacing:.3px}
.sub{color:var(--muted);font-size:12.5px;margin-top:3px}
.spacer{flex:1}
.pill{display:inline-flex;align-items:center;gap:7px;padding:8px 15px;border-radius:999px;
  background:#fff;border:1px solid var(--line);box-shadow:var(--shadow);font-size:13.5px;font-weight:600}
.dot{width:8px;height:8px;border-radius:50%;background:var(--muted)}
.dot.on{background:var(--ok);box-shadow:0 0 0 4px rgba(95,174,139,.16)}
.dot.off{background:var(--bad);box-shadow:0 0 0 4px rgba(217,115,107,.14)}
#clock{color:var(--muted);font-size:12.5px;font-variant-numeric:tabular-nums}

/* ---------- 能力链 ---------- */
.stack{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 16px}
.chip{display:inline-flex;align-items:center;gap:8px;background:#fff;border:1px solid var(--line);
  border-radius:999px;padding:7px 14px;font-size:12.5px;box-shadow:0 3px 12px rgba(160,130,110,.07)}
.chip .d{width:7px;height:7px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 3px rgba(95,174,139,.14)}
.chip.off{color:#B9AFA6}
.chip.off .d{background:#DDD2C8;box-shadow:none}
.chip b{font-weight:650}

/* ---------- 性能看板 ---------- */
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:16px 18px;box-shadow:var(--shadow)}
.kpi .k{font-size:12px;color:var(--muted);letter-spacing:.4px}
.kpi .v{font-size:27px;font-weight:700;margin-top:6px;font-variant-numeric:tabular-nums;line-height:1.15}
.kpi .v small{font-size:13px;font-weight:600;color:var(--muted);margin-left:3px}
.kpi .h{font-size:11.5px;color:var(--muted);margin-top:5px;min-height:14px}
.kpi.hl{background:linear-gradient(160deg,#FFF3EC,#FFFFFF);border-color:#F7DFD2}
.kpi.hl .v{color:var(--accent)}

/* ---------- 布局 ---------- */
.grid{display:grid;grid-template-columns:1.05fr .95fr;gap:16px;align-items:start}
.card{background:var(--card);border:1px solid var(--line);border-radius:22px;padding:18px 20px;box-shadow:var(--shadow);margin-bottom:16px}
.card h2{font-size:12.5px;margin:0 0 14px;color:var(--muted);font-weight:600;letter-spacing:1.2px}
.snap{width:100%;border-radius:16px;display:block;background:#F2EAE3;min-height:120px;object-fit:cover}
.row{display:flex;justify-content:space-between;gap:14px;padding:8px 0;border-bottom:1px solid var(--line);font-size:14px}
.row:last-child{border:0}
.row .lbl{color:var(--muted);flex:0 0 auto}
.row .val{text-align:right;word-break:break-all}
.dim{color:var(--muted)}
.ok{color:var(--ok)} .warn{color:var(--warn)} .bad{color:var(--bad)}
.answer{font-size:16px;line-height:1.65;margin:2px 0 12px;color:#2F2C29}

/* 时延条 */
.bar{display:flex;height:13px;border-radius:8px;overflow:hidden;background:#F5EDE6;margin:10px 0 8px}
.bar i{display:block;height:100%}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--muted)}
.legend b{font-weight:600;color:var(--ink);font-variant-numeric:tabular-nums}
.sw{display:inline-block;width:8px;height:8px;border-radius:3px;margin-right:5px;vertical-align:1px}
.badge{display:inline-block;padding:3px 9px;border-radius:999px;background:var(--accent-soft);color:#B5604B;font-size:11.5px;font-weight:600}

/* 按钮 */
button{background:#fff;color:var(--ink);border:1px solid var(--line);border-radius:14px;
  padding:10px 15px;font-size:13.5px;cursor:pointer;margin:0 8px 8px 0;
  box-shadow:0 3px 10px rgba(160,130,110,.08);transition:.16s}
button:hover{border-color:#EBC7B6;transform:translateY(-1px)}
button:active{transform:translateY(0)}
button.fall{background:#FDF0EC;border-color:#F3D3C8;color:#B5604B}
input{background:#FFFDFB;border:1px solid var(--line);color:var(--ink);border-radius:12px;padding:9px 12px;font-size:13.5px;width:100%}
input:focus{outline:none;border-color:#EBC7B6}

/* 事件流 */
.ev{font-size:13px;padding:7px 0;border-bottom:1px dashed var(--line);display:flex;gap:9px;align-items:baseline}
.ev:last-child{border:0}
.ev .t{color:var(--muted);white-space:nowrap;font-variant-numeric:tabular-nums;font-size:12px}
.tag{font-size:11px;padding:2px 7px;border-radius:7px;background:#F7F1EA;color:var(--muted);white-space:nowrap}
.scroll{max-height:320px;overflow:auto}
footer{color:var(--muted);font-size:12px;text-align:center;padding:8px 0 0}
footer .q{color:#B9AFA6;font-style:italic;margin-bottom:4px}
@media(max-width:900px){
  .kpis{grid-template-columns:repeat(2,1fr)}
  .grid{grid-template-columns:1fr}
  header{flex-wrap:wrap}
}
</style></head><body>
<div class="wrap">

<header>
  <div class="face"><span class="eye l"></span><span class="eye r"></span><span class="mouth"></span></div>
  <div>
    <h1>大白 Baymax</h1>
    <div class="sub">基于 RISC-V K1 的多模态智能健康陪护终端</div>
  </div>
  <div class="spacer"></div>
  <div class="pill"><span class="dot" id="dot"></span><span id="status">连接中</span></div>
  <div id="clock"></div>
</header>

<div class="stack" id="stack"></div>

<div class="kpis">
  <div class="kpi hl"><div class="k">唤醒 → 出声</div><div class="v" id="k_wake">待测<small>ms</small></div><div class="h">KWS 命中到播放器启动</div></div>
  <div class="kpi"><div class="k">端到端平均</div><div class="v" id="k_avg">—<small>s</small></div><div class="h" id="k_avg_h">最近 30 轮</div></div>
  <div class="kpi"><div class="k">语音缓存命中率</div><div class="v" id="k_hit">—<small>%</small></div><div class="h" id="k_hit_h">已命中 / 总轮次</div></div>
  <div class="kpi"><div class="k">交互轮次</div><div class="v" id="k_cnt">—</div><div class="h" id="k_cnt_h">本地推理，数据不出户</div></div>
</div>

<div class="grid">
  <div>
    <div class="card">
      <h2>最近一轮</h2>
      <div class="dim" style="font-size:12.5px;margin-bottom:6px" id="q">还没有对话</div>
      <div class="answer" id="a">—</div>
      <div class="bar" id="bar"><i style="width:0;background:#E8836A"></i></div>
      <div class="legend" id="legend"><span class="dim">时延明细会在下一轮对话后出现</span></div>
    </div>

    <div class="card">
      <h2>演示控制</h2>
      <button class="fall" onclick="cmd({cmd:'fall'})">模拟跌倒</button>
      <button onclick="cmd({cmd:'measure_hr'})">测心率血氧</button>
      <button onclick="cmd({cmd:'say',text:'我在呢，您说。'})">唤醒应答</button>
      <button onclick="cmd({cmd:'say',text:'该吃药了，记得按医生说的剂量吃。'})">用药提醒</button>
      <button onclick="cmd({cmd:'stop'})">停止播报</button>
      <div style="margin-top:12px;display:flex;gap:9px">
        <input id="saytext" placeholder="让它说一句（需已缓存）">
        <button style="margin:0;white-space:nowrap" onclick="cmd({cmd:'say',text:document.getElementById('saytext').value})">发送</button>
      </div>
    </div>

    <div class="card">
      <h2>硬件状态</h2>
      <div id="hw"></div>
      <div class="dim" style="font-size:11.5px;margin-top:8px">标「模拟」= 没检测到该硬件，用同一套算法跑仿真数据，界面上不冒充真实读数。</div>
    </div>

    <div class="card">
      <h2>现场画面（CSI 摄像头）</h2>
      <img id="snap" class="snap" alt="点「拍一张」抓拍现场">
      <div style="margin-top:10px;display:flex;gap:8px;align-items:center">
        <button style="margin:0" onclick="snapLive()">拍一张</button>
        <span class="dim" style="font-size:12px" id="snapinfo">抓拍约需 3 秒</span>
      </div>
    </div>
  </div>

  <div>
    <div class="card">
      <h2>健康档案</h2>
      <div id="events" class="scroll"></div>
    </div>
    <div class="card">
      <h2>健康数据</h2>
      <div id="vitals"></div>
    </div>
    <div class="card">
      <h2>用药提醒</h2>
      <div id="reminders"></div>
      <div style="margin-top:10px;display:flex;gap:8px">
        <input id="rtime" placeholder="08:00" style="width:92px">
        <input id="rtext" placeholder="提醒内容">
        <button style="margin:0;white-space:nowrap" onclick="cmd({cmd:'add_reminder',time:document.getElementById('rtime').value,text:document.getElementById('rtext').value})">添加</button>
      </div>
    </div>
  </div>
</div>

<footer>
  <div class="q">"I will always be with you."</div>
  大白 Baymax · RISC-V K1 · 端侧推理，语音与健康数据都不出户
</footer>
</div>

<script>
const $ = id => document.getElementById(id);
function esc(s){return String(s==null?'':s).replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))}
function cmd(o){fetch('/api/cmd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(o)}).then(()=>setTimeout(refresh,600))}
function num(v,d){return (typeof v==='number'&&isFinite(v))?v.toFixed(d):'—'}

function renderLatency(lat){
  const L = [['录音',lat.record,'#E8836A'],['识别',lat.asr,'#E7A98F'],['检索',lat.retrieve,'#EFC9B6']];
  const total = L.reduce((s,x)=>s+(x[1]||0),0);
  const bar = $('bar'); bar.innerHTML='';
  if(!lat || !lat.total){ $('legend').innerHTML='<span class="dim">时延明细会在下一轮对话后出现</span>'; return; }
  if(total<=0){ $('legend').innerHTML='<span class="dim">本轮只有总时延（旧记录），下一轮会有分项</span>'; return; }
  L.forEach(([name,v,color])=>{
    const w = total>0 ? Math.max(2,(v||0)/total*100) : 0;
    const i=document.createElement('i'); i.style.width=w+'%'; i.style.background=color; bar.appendChild(i);
  });
  const cache = (typeof lat.cache==='number' && lat.cache===0);
  $('legend').innerHTML = L.map(([n,v,c])=>'<span><i class="sw" style="background:'+c+'"></i>'+n+' <b>'+num(v,2)+'s</b></span>').join('')
    + '<span><i class="sw" style="background:#5FAE8B"></i>播报 <b>'+(cache?'缓存直出':'现场生成')+'</b></span>'
    + (lat.mode? '<span class="badge">'+esc(lat.mode)+'</span>':'');
}

function refresh(){
  fetch('/api/state').then(r=>r.json()).then(d=>{
    const s=d.state||{}, m=s.metrics||{};
    $('clock').textContent=d.now||'';
    $('dot').className='dot '+(s.online?'on':'off');
    $('status').textContent=s.online?'在线守护中':'主程序未连接';

    // KPI
    $('k_wake').innerHTML = (m.wake_ms!=null? m.wake_ms : '待测')+'<small>ms</small>';
    const turns=(d.events||[]).filter(e=>e.kind==='turn'&&typeof e.latency==='number').slice(-30);
    const lats=turns.map(e=>e.latency);
    const avg=lats.length? lats.reduce((a,b)=>a+b,0)/lats.length : null;
    $('k_avg').innerHTML = (avg!=null? avg.toFixed(2):'待测')+'<small>s</small>';
    if(lats.length){
      const mn=Math.min(...lats), mx=Math.max(...lats);
      $('k_avg_h').textContent='最快 '+(mn<0.005?'<0.01':mn.toFixed(2))+'s · 最慢 '+mx.toFixed(2)+'s';
    } else {
      $('k_avg_h').textContent='对它说「你好大白」开始第一轮';
    }
    const allTurns=(d.events||[]).filter(e=>e.kind==='turn');
    const hitTurns=allTurns.filter(e=>e.cached===true).length;
    const cnt=Number(s.turn_count||0);
    $('k_hit').innerHTML = (allTurns.length? Math.round(hitTurns/allTurns.length*100):'待测')+'<small>%</small>';
    $('k_hit_h').textContent = allTurns.length? (hitTurns+' / '+allTurns.length+' 轮命中预生成语音') : '已命中 / 总轮次';
    $('k_cnt').textContent = cnt||0;
    $('k_cnt_h').textContent = d.cache? ('语音缓存：'+d.cache) : '本地推理，数据不出户';

    // 能力链
    const sk=d.stack||{};
    const chips=[
      ['语音唤醒 KWS', s.online],
      ['语音识别 ASR', s.online],
      ['语音合成 TTS', !!(d.cache||'').match(/\\d+/)],
      ['本地大模型 Qwen2.5', !!sk.llm],
      ['健康知识库 '+(sk.kb||0)+' 条', (sk.kb||0)>0],
      ['端侧推理 · 不出户', true]
    ];
    $('stack').innerHTML = chips.map(([n,on])=>
      '<span class="chip'+(on?'':' off')+'"><i class="d"></i>'+n+'</span>').join('');

    $('q').textContent = s.last_question? ('“'+s.last_question+'”') : '还没有对话';
    $('a').textContent = s.last_answer || '—';
    renderLatency(s.last_latency||{});

    // 事件流
    const ev=$('events');
    const kindName={turn:'对话',mood:'关怀',fall:'跌倒',vitals:'体征',alert:'告警',reminder:'提醒'};
    ev.innerHTML=(d.events||[]).slice().reverse().slice(0,40).map(e=>
      '<div class="ev"><span class="t">'+esc((e.ts||'').slice(11))+'</span>'+
      '<span class="tag">'+esc(kindName[e.kind]||e.kind||'')+'</span>'+
      '<span>'+esc(e.question||e.text||e.title||e.answer||'')+'</span></div>').join('')||'<div class="dim">暂无记录</div>';

    // 提醒
    $('reminders').innerHTML=(d.reminders||[]).map(r=>
      '<div class="row"><span class="lbl">'+esc(r.time)+'</span><span class="val dim">'+esc(r.text)+'</span></div>').join('')||'<div class="dim">暂无提醒</div>';

    // 健康数据
    const vt=d.vitals||{}, st=vt.stats||{}, names={bp:'血压',glucose:'血糖',pulse:'心率',spo2:'血氧',temp:'体温',weight:'体重'};
    const stHtml=Object.keys(st).map(k=>{
      const v=st[k], mainV=(k==='bp'? v.avg_sys : v.avg);
      return '<div class="row"><span class="lbl">'+esc(names[k]||k)+'</span><span class="val dim">共 '+v.count+' 次 · 均值 '+mainV+'</span></div>';
    }).join('');
    const recHtml=(vt.recent||[]).slice(0,6).map(r=>
      '<div class="ev"><span class="t">'+esc((r.ts||'').slice(5,16))+'</span>'+
      '<span class="tag">'+esc(names[r.kind]||r.kind)+'</span><span>'+esc(r.text)+'</span></div>').join('');
    $('vitals').innerHTML = (stHtml+recHtml) || '<div class="dim">还没有数据，对着大白说「我血压 150 90」试试</div>';

    // 硬件
    try{
      const hw=d.hw||{}, devs=hw.devices||[];
      const colmap={'在线':'ok','模拟':'warn','缺失':'bad','未探测':'dim'};
      const hwHtml=devs.map(r=>{
        const cls=colmap[r.state]||'dim';
        return '<div class="row"><span class="lbl">'+esc(r.dev)+' <span class="dim">'+esc(r.model)+'</span></span>'+
               '<span class="val '+cls+'">'+esc(r.state)+'</span></div>'+
               '<div class="dim" style="font-size:11px;margin:-4px 0 6px">'+esc(r.detail)+'</div>';
      }).join('');
      $('hw').innerHTML=(hwHtml||'<div class="dim">主程序还没写硬件状态，先启动 ./start_baymax3.sh</div>')+
        (hw.ts? '<div class="dim" style="font-size:11px;margin-top:6px">更新于 '+esc(hw.ts)+'</div>':'');
    }catch(e){ $('hw').textContent='硬件状态渲染失败：'+e }
  }).catch(()=>{ $('dot').className='dot off'; $('status').textContent='面板未连上主程序' });
}
setInterval(refresh,2000); refresh();

function snapLive(){
  var img=$('snap'), info=$('snapinfo');
  info.textContent='正在抓拍…';
  img.src='/snapshot.jpg?live=1&t='+Date.now();
  img.onload=function(){ info.textContent='刚刚拍下 · '+new Date().toLocaleTimeString(); };
  img.onerror=function(){ info.textContent='抓拍失败（摄像头忙或未就绪）'; };
}
</script>
<div style="max-width:1180px;margin:6px auto 24px;padding:14px 18px;border-radius:16px;
            background:#FBF6F0;border:1px solid var(--line);color:#7A6E64;font-size:12.5px;line-height:1.7">
  <b>免责声明</b>：大白是健康陪护辅助工具，可提供健康知识、数据记录与紧急通知，
  <b>但不能代替医生诊断，也不能代替拨打 120</b>。出现急症请立即就医或拨打急救电话。
</div>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except OSError:
            pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path.startswith("/snapshot.jpg"):
            return self._snapshot()
        if self.path.startswith("/api/state"):
            state = svc._read_json(svc.STATE_PATH, {})
            # 超过 90 秒没心跳就当离线
            online = False
            try:
                hb = time.mktime(time.strptime(state.get("hb") or state.get("ts"),
                                               "%Y-%m-%d %H:%M:%S"))
                online = (time.time() - hb) < 90
            except Exception:
                online = False
            state["online"] = online
            payload = {
                "now": time.strftime("%H:%M:%S"),
                "state": state,
                "events": svc.load_events(80),
                "reminders": svc.load_reminders(),
                "vitals": (health.summary() if health else {}),
                "cache": self._cache_summary(),
                "hw": self._hw(),
                "stack": self._stack(),
            }
            return self._send(200, json.dumps(payload, ensure_ascii=False))
        return self._send(404, '{"error":"not found"}')

    def _snapshot(self):
        """抓一张 CSI 摄像头画面给网页显示（?live=1 时现场抓，否则用缓存）。"""
        path = os.path.join(svc.HOME_DIR, "snapshot.jpg")
        live = "live=1" in self.path
        if live or not os.path.isfile(path):
            try:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                import baymax_sensors as _s
                if not _s.csi_snapshot(path):
                    return self._send(503, b"", "image/jpeg")
            except Exception:
                return self._send(503, b"", "image/jpeg")
        try:
            with open(path, "rb") as f:
                return self._send(200, f.read(), "image/jpeg")
        except OSError:
            return self._send(404, b"", "image/jpeg")

    def do_POST(self):
        if self.path.startswith("/api/cmd"):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n).decode("utf-8") if n else "{}"
                obj = json.loads(body)
                if not isinstance(obj, dict):
                    raise ValueError
            except Exception:
                return self._send(400, '{"ok":false}')
            svc._write_json(svc.CMD_PATH, obj)
            return self._send(200, '{"ok":true}')
        return self._send(404, '{"error":"not found"}')

    def _hw(self):
        """硬件状态由主程序写到 ~/.baymax/hw.json，面板只读，避免重复探测。"""
        try:
            with open(os.path.join(svc.HOME_DIR, "hw.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _stack(self):
        """能力链状态：本地大模型是否在线、知识库条数。10 秒缓存，避免拖慢页面。"""
        global _STACK_CACHE
        now = time.time()
        if now - _STACK_CACHE.get("ts", 0) < 10:
            return _STACK_CACHE.get("data", {})
        llm = False
        try:
            import urllib.request
            with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=1.2) as r:
                llm = (r.status == 200)
        except Exception:
            llm = False
        kb = 0
        try:
            with open(os.path.expanduser("~/Baymax_v14/medical_knowledge.jsonl"),
                      "r", encoding="utf-8") as f:
                kb = sum(1 for line in f if line.strip())
        except Exception:
            kb = 0
        data = {"llm": llm, "kb": kb}
        _STACK_CACHE = {"ts": now, "data": data}
        return data

    def _cache_summary(self):
        root = os.path.expanduser("~/voice_cache_staging_v2")
        out = []
        try:
            n = len(os.listdir(os.path.join(root, "voice_cache_payload_v2", "common_wav")))
            out.append("常用语 %d 条" % n)
        except OSError:
            pass
        try:
            n = len(os.listdir(os.path.join(root, "voice_cache_payload_v2", "medical_wav")))
            out.append("知识全文 %d 条" % n)
        except OSError:
            pass
        return " · ".join(out)

    def log_message(self, fmt, *args):
        pass


def main():
    svc._ensure_dir()
    svc.load_reminders()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("大白面板已启动： http://<K1的IP>:%d" % PORT)
    print("本机自测： http://127.0.0.1:%d" % PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n面板退出")


if __name__ == "__main__":
    main()
