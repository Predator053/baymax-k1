#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""大白 3.0 服务层：健康档案 / 用药提醒 / 网页面板指令通道。

设计要点：主程序只负责"说话"，这里负责"记事和提醒"。
两者通过 ~/.baymax/ 下的三个文件通信，不需要 socket 或额外依赖：

  state.json     主程序写入：当前状态 + 最近一轮对话（面板只读）
  events.jsonl   追加写：每一次交互、跌倒、提醒、测量（健康档案）
  reminders.json 用药提醒计划（面板可增删，主程序定时读取）
  cmd.json       面板写的指令（模拟跌倒 / 让它说一句 / 加提醒）

这样即使面板进程挂了，语音主链路也不受影响。
"""
import json
import os
import threading
import time

HOME_DIR = os.getenv("BAYMAX_HOME") or os.path.expanduser("~/.baymax")
STATE_PATH = os.path.join(HOME_DIR, "state.json")
EVENTS_PATH = os.path.join(HOME_DIR, "events.jsonl")
REMINDERS_PATH = os.path.join(HOME_DIR, "reminders.json")
CMD_PATH = os.path.join(HOME_DIR, "cmd.json")

_DEFAULT_REMINDERS = [
    {"id": "morning", "time": "08:00", "text": "该吃药了，记得按医生说的剂量吃。", "enabled": True},
    {"id": "noon", "time": "12:30", "text": "该吃药了，记得按医生说的剂量吃。", "enabled": True},
    {"id": "evening", "time": "19:00", "text": "该吃药了，记得按医生说的剂量吃。", "enabled": True},
]


def _ensure_dir():
    os.makedirs(HOME_DIR, exist_ok=True)


def _read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, type(default)) else default
    except Exception:
        return default


def _write_json(path, data):
    """原子写：先写临时文件再 rename，避免面板读到半截 JSON。"""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        pass


def append_event(kind, **fields):
    """往健康档案追加一条事件。"""
    _ensure_dir()
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind}
    rec.update(fields)
    try:
        with open(EVENTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return rec


def load_events(limit=200):
    _ensure_dir()
    items = []
    try:
        with open(EVENTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return items[-limit:]


def load_reminders():
    _ensure_dir()
    if not os.path.isfile(REMINDERS_PATH):
        _write_json(REMINDERS_PATH, {"items": _DEFAULT_REMINDERS})
    data = _read_json(REMINDERS_PATH, {})
    items = data.get("items")
    return items if isinstance(items, list) else []


def save_reminders(items):
    _ensure_dir()
    _write_json(REMINDERS_PATH, {"items": items})


class Service:
    """主程序里跑的两个后台线程：用药提醒 + 面板指令。"""

    def __init__(self, say=None, on_fall=None, on_stop=None, stop_event=None):
        self.say = say or (lambda text: False)
        self.on_fall = on_fall or (lambda g=1.0: None)
        self.on_stop = on_stop or (lambda: None)
        self.stop_event = stop_event or threading.Event()
        self._fired = {}
        self._threads = []
        _ensure_dir()

    # ---------- 对外：让面板看到当前状态 ----------
    def note_turn(self, question="", answer="", latency=None, mode=""):
        state = _read_json(STATE_PATH, {})
        state.update({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "online": True,
            "mode": mode or state.get("mode", ""),
            "last_question": question,
            "last_answer": answer,
            "last_latency": latency or {},
            "turn_count": int(state.get("turn_count", 0)) + 1,
        })
        _write_json(STATE_PATH, state)

    def note_event(self, kind, **fields):
        rec = append_event(kind, **fields)
        state = _read_json(STATE_PATH, {})
        state["last_event"] = rec
        state["last_event_at"] = rec["ts"]
        _write_json(STATE_PATH, state)

    def note_metric(self, key, value):
        """记录一条性能指标（面板展示用），例如唤醒→出声的毫秒数。"""
        state = _read_json(STATE_PATH, {})
        metrics = state.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        metrics[str(key)] = value
        state["metrics"] = metrics
        _write_json(STATE_PATH, state)

    def bump_metric(self, key, delta=1):
        """计数器累加，例如缓存命中次数。"""
        state = _read_json(STATE_PATH, {})
        metrics = state.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        try:
            metrics[str(key)] = int(metrics.get(str(key), 0)) + int(delta)
        except Exception:
            metrics[str(key)] = int(delta)
        state["metrics"] = metrics
        _write_json(STATE_PATH, state)

    # ---------- 后台线程 ----------
    def start(self):
        for target, name in ((self._reminder_loop, "Reminder"), (self._command_loop, "PanelCmd")):
            th = threading.Thread(target=target, name=name, daemon=True)
            th.start()
            self._threads.append(th)

    def _reminder_loop(self):
        while not self.stop_event.is_set():
            try:
                now = time.localtime()
                today = time.strftime("%Y-%m-%d", now)
                hhmm = time.strftime("%H:%M", now)
                for item in load_reminders():
                    if not isinstance(item, dict) or not item.get("enabled", True):
                        continue
                    rid = str(item.get("id") or item.get("time"))
                    if str(item.get("time")) != hhmm or self._fired.get(rid) == today:
                        continue
                    text = item.get("text") or "该吃药了，记得按医生说的剂量吃。"
                    self._fired[rid] = today
                    print(f"\n⏰ [用药提醒] {text}")
                    self.say(text)
                    self.note_event("reminder", reminder_id=rid, text=text)
            except Exception:
                pass
            self.stop_event.wait(20.0)

    def _command_loop(self):
        last_report = 0.0
        while not self.stop_event.is_set():
            try:
                cmd = _read_json(CMD_PATH, {})
                if isinstance(cmd, dict) and cmd.get("cmd"):
                    try:
                        os.remove(CMD_PATH)
                    except OSError:
                        pass
                    self._dispatch(cmd)
            except Exception:
                pass
            # 每 15 秒刷新一次面板状态（存活信号）
            if time.monotonic() - last_report > 15:
                last_report = time.monotonic()
                state = _read_json(STATE_PATH, {})
                state["online"] = True
                state["hb"] = time.strftime("%Y-%m-%d %H:%M:%S")
                _write_json(STATE_PATH, state)
            self.stop_event.wait(1.0)

    def _sensors(self):
        """硬件模块是可选依赖：拿不到就返回 None，绝不因为面板按钮崩主程序。"""
        try:
            import baymax_sensors as s
            return s
        except Exception:
            return None

    def _dispatch(self, cmd):
        name = str(cmd.get("cmd"))
        if name == "fall":
            # 优先让传感器回放一次跌倒波形，这样走的是完整的"数据 → 状态机 → 关怀"链路；
            # 真机接了 MPU6050 时 inject 不生效，退化成直接触发事件。
            s = self._sensors()
            via_sensor = False
            if s is not None:
                try:
                    via_sensor = bool(s.simulate_fall())
                except Exception:
                    via_sensor = False
            if via_sensor:
                print("\n🚨 [演示] 面板触发跌倒 → 模拟波形回放中")
            else:
                print("\n🚨 [演示] 面板触发跌倒事件（直接触发）")
                self.on_fall(1.0)
            self.note_event("fall", source="panel", manual=True, sensor_sim=via_sensor)
        elif name in ("measure_hr", "measure", "vital_measure"):
            s = self._sensors()
            ok = False
            if s is not None:
                try:
                    ok = bool(s.measure_health())
                except Exception:
                    ok = False
            print("\n💓 [演示] 面板请求测量心率血氧：%s" % ("已受理" if ok else "传感器未就绪"))
            self.note_event("measure_request", ok=ok)
        elif name == "say":
            text = str(cmd.get("text") or "").strip()
            if text:
                self.say(text)
                self.note_event("panel_say", text=text)
        elif name == "stop":
            self.on_stop()
        elif name == "add_reminder":
            items = load_reminders()
            items.append({
                "id": "r%d" % int(time.time()),
                "time": str(cmd.get("time") or "08:00"),
                "text": str(cmd.get("text") or "该吃药了，记得按医生说的剂量吃。"),
                "enabled": True,
            })
            save_reminders(items)
            self.note_event("reminder_added", time=cmd.get("time"))
