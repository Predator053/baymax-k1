#!/bin/bash
# 演示前自检（新版：USB 麦克风 + 蓝牙音响）
cd /home/baymax/Baymax_v14 || exit 1
PASS=0; FAIL=0
ok(){ echo "  ✅ $1"; PASS=$((PASS+1)); }
no(){ echo "  ❌ $1"; FAIL=$((FAIL+1)); }

BT_MAC="${BAYMAX_BT_MAC:-41:42:C3:8E:D2:ED}"
BT_SINK="bluez_output.$(echo "$BT_MAC" | tr ':' '_').1"

echo "══════════ 大白演示自检 $(date '+%H:%M:%S') ══════════"

echo "【1】进程"
pgrep -f 'python3 -u baymax_main.py' >/dev/null && ok "主程序在跑" || no "主程序没跑"
pgrep -f 'baymax_panel.py' >/dev/null && ok "面板在跑" || no "面板没跑"
pgrep -f arecord >/dev/null && ok "唤醒监听在跑" || no "唤醒监听没跑"

echo "【2】输入（USB 麦克风）"
arecord -l 2>/dev/null | grep -qiE "USB" && ok "USB 麦克风已识别" || no "USB 麦克风没识别"

echo "【3】输出（蓝牙音响）"
bluetoothctl info "$BT_MAC" 2>/dev/null | grep -q "Connected: yes" && ok "蓝牙音响已连接" || no "蓝牙音响未连接"
pactl list short sinks 2>/dev/null | grep -q "$BT_SINK" && ok "蓝牙 sink 已建立" || no "蓝牙 sink 不存在"
pactl get-default-sink 2>/dev/null | grep -q "bluez" && ok "默认输出指向蓝牙" || no "默认输出不是蓝牙"
python3 -c "
import wave,struct,math
sr=48000
with wave.open('/tmp/_chk.wav','wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes(b''.join(struct.pack('<hh',*(2*[int(8000*math.sin(2*math.pi*800*t/sr))])) for t in range(sr//4)))
" 2>/dev/null
aplay -q -D default /tmp/_chk.wav >/dev/null 2>&1 && ok "音响能出声" || no "音响放不出来"

echo "【4】语音缓存"
python3 - <<'PY'
import json, os
root = "/home/baymax/voice_cache_staging_v2/voice_cache_payload_v2"
c = json.load(open(os.path.join(root, "common_manifest.json"), encoding="utf-8"))
m = json.load(open(os.path.join(root, "manifest.json"), encoding="utf-8"))
print("  ✅ 常用语 %d 条 / 医疗全文 %d 条" % (len(c.get("records", {})), len(m.get("records", {}))))
PY

echo "【5】三类问答分流"
python3 - <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("bm", "/home/baymax/Baymax_v14/baymax_main.py")
mod = importlib.util.module_from_spec(spec); sys.modules["bm"] = mod
try: spec.loader.exec_module(mod)
except SystemExit: pass
cases = [("什么是高血压","高血压是什么"),("为什么会有高血压","高血压是怎么来的"),("有了高血压应该怎么办","高血压日常管理")]
bad = 0
for q, expect in cases:
    r = mod.retrieve_medical_knowledge(q, top_k=1)
    title = r[0]["item"].get("title") if r else "未命中"
    if title != expect: bad += 1
print("  ✅ 3/3 分流正确" if bad == 0 else "  ❌ %d 条分流错误" % bad)
PY

echo "【6】血压解析与分级"
python3 - <<'PY'
import sys; sys.path.insert(0, '.')
import baymax_health as h
bad = 0
for text, exp in [("我血压一百五九十","150/90"),("血压一百八十比一百一","180/110")]:
    v = h.parse_vitals(text)
    if not v or v.get("text") != exp: bad += 1
print("  ✅ 中文数字解析正常" if bad == 0 else "  ❌ 解析异常")
PY

echo "【6.5】关键语句语音缓存（没缓存=现场没有声音）"
python3 - <<'PY' 2>&1 | grep -vE '^\[' | head -10
import importlib.util, sys, os
spec = importlib.util.spec_from_file_location("bm", "/home/baymax/Baymax_v14/baymax_main.py")
mod = importlib.util.module_from_spec(spec); sys.modules["bm"] = mod
try: spec.loader.exec_module(mod)
except SystemExit: pass
items = [("跌倒询问", mod.FALL_ASK), ("跌倒确认", mod.FALL_CONFIRM),
         ("跌倒取消", mod.FALL_CANCEL), ("无应答", mod.FALL_NO_ANSWER),
         ("关怀-主动", mod.MOOD_CHECK), ("关怀-安抚", mod.MOOD_ASK),
         ("关怀-安慰", mod.MOOD_COMFORT), ("关怀-收尾", mod.MOOD_BYE)]
miss = [n for n, t in items if not mod.resolve_any_voice_cache(t)]
if miss:
    print("  ❌ 缺语音：%s（现场会没声音！）" % "、".join(miss))
else:
    print("  ✅ 8/8 关键语句都有缓存")
PY


echo "【7】硬件"
python3 -c "
import json,os
d=json.load(open(os.path.expanduser('~/.baymax/hw.json'),encoding='utf-8'))
for r in d.get('devices',[]): print('  · %-9s %s' % (r.get('dev'), r.get('state')))
" 2>/dev/null

echo "【8】4G 与网络"
timeout 15 mmcli -L 2>/dev/null | grep -q Modem && echo "  ✅ 4G 模块在线" || echo "  ❌ 4G 模块不在"
ping -c1 -W2 223.5.5.5 >/dev/null 2>&1 && echo "  ✅ 外网可达" || echo "  ⚠️ 外网不通（短信仍可用）"

echo
echo "══════════ 结果：$PASS 项通过，$FAIL 项异常 ══════════"
