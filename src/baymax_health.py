#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import re
import time

HOME_DIR = os.getenv("BAYMAX_HOME") or os.path.expanduser("~/.baymax")
VITALS_PATH = os.path.join(HOME_DIR, "vitals.jsonl")

_FULLWIDTH = str.maketrans("０１２３４５６７８９／．", "0123456789/.")
_UNITS = r"(?:毫米汞柱|mmhg|毫升|mg|次|度|℃)?"


def _norm(text):
    return str(text or "").translate(_FULLWIDTH)


_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_NUM_RE = re.compile(
    r"([一二两三四五六七八九])?百([一二三四五六七八九])?十([一二三四五六七八九])?"
    r"|([一二两三四五六七八九])?百([一二三四五六七八九])?"
    r"|([一二两三四五六七八九])?十([一二三四五六七八九])?"
    r"|([零〇一二两三四五六七八九])"
)


def _cn_to_arabic(text):
    def repl(m):
        if m.group(1) is not None or m.group(2) is not None or m.group(3) is not None:
            hundreds = _CN_DIGIT.get(m.group(1) or "一", 1) * 100
            tens = _CN_DIGIT.get(m.group(2), 0) * 10 if m.group(2) else 0
            ones = _CN_DIGIT.get(m.group(3), 0) if m.group(3) else 0
            return " %d " % (hundreds + tens + ones)
        if m.group(4) is not None or m.group(5) is not None:
            hundreds = _CN_DIGIT.get(m.group(4) or "一", 1) * 100
            tens = _CN_DIGIT.get(m.group(5), 0) * 10 if m.group(5) else 0
            return " %d " % (hundreds + tens)
        if m.group(6) is not None or m.group(7) is not None:
            tens = _CN_DIGIT.get(m.group(6) or "一", 1) * 10
            ones = _CN_DIGIT.get(m.group(7), 0) if m.group(7) else 0
            return " %d " % (tens + ones)
        if m.group(8) is not None:
            return " %d " % _CN_DIGIT[m.group(8)]
        return m.group(0)

    out = text.replace("点", ".")
    out = _CN_NUM_RE.sub(repl, out)
    out = re.sub(r"\s*\.\s*", ".", out)
    return re.sub(r"\s+", " ", out).strip()


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_vitals(text):
    t = _norm(text)
    if not t:
        return None
    t = _cn_to_arabic(t)

    m = re.search(r"(?:血压|高压|血压是)?\s*(\d{2,3})\s*[/比比之于]\s*(\d{2,3})", t)
    if m:
        sys_v, dia_v = _num(m.group(1)), _num(m.group(2))
        if sys_v and dia_v and 50 <= dia_v <= 200 and 60 <= sys_v <= 300 and sys_v > dia_v:
            return {"kind": "bp", "sys": int(sys_v), "dia": int(dia_v),
                    "text": "%d/%d" % (sys_v, dia_v)}

    m = re.search(r"高压\s*(\d{2,3})", t)
    m2 = re.search(r"低压\s*(\d{2,3})", t)
    if m and m2:
        sys_v, dia_v = _num(m.group(1)), _num(m2.group(1))
        if sys_v and dia_v and 40 <= dia_v <= 200 and 60 <= sys_v <= 300:
            return {"kind": "bp", "sys": int(sys_v), "dia": int(dia_v),
                    "text": "%d/%d" % (sys_v, dia_v)}

    m = re.search(
        r"(?:血压|高压)\s*(?:是|有|为|高|低|有点|大概)?\s*"
        r"(\d{2,3})\s*[，,、\s比之于]*\s*(\d{2,3})", t)
    if m:
        sys_v, dia_v = _num(m.group(1)), _num(m.group(2))
        if sys_v and dia_v and 40 <= dia_v <= 200 and 60 <= sys_v <= 300 and sys_v > dia_v:
            return {"kind": "bp", "sys": int(sys_v), "dia": int(dia_v),
                    "text": "%d/%d" % (sys_v, dia_v)}

    m = re.search(r"血糖\s*(?:是|有|为)?\s*(\d{1,2}(?:\.\d)?)", t)
    if m:
        v = _num(m.group(1))
        if v is not None and 1.0 <= v <= 40.0:
            return {"kind": "glucose", "value": v, "text": "%.1f" % v}

    m = re.search(r"(?:心率|脉搏|心跳)\s*(?:是|有|为)?\s*(\d{2,3})", t)
    if m:
        v = _num(m.group(1))
        if v is not None and 25 <= v <= 250:
            return {"kind": "pulse", "value": int(v), "text": "%d" % v}

    m = re.search(r"(?:血氧|氧饱和度)\s*(?:是|有|为)?\s*(\d{2,3})", t)
    if m:
        v = _num(m.group(1))
        if v is not None and 50 <= v <= 100:
            return {"kind": "spo2", "value": int(v), "text": "%d" % v}

    m = re.search(r"体温\s*(?:是|有|为)?\s*(3\d(?:\.\d)?)", t)
    if m:
        v = _num(m.group(1))
        if v is not None and 33.0 <= v <= 43.0:
            return {"kind": "temp", "value": v, "text": "%.1f" % v}

    m = re.search(r"体重\s*(?:是|有|为)?\s*(\d{2,3}(?:\.\d)?)", t)
    if m:
        v = _num(m.group(1))
        if v is not None and 20.0 <= v <= 200.0:
            return {"kind": "weight", "value": v, "text": "%.1f" % v}

    return None


def bp_grade(sys_v, dia_v):
    if sys_v >= 180 or dia_v >= 110:
        return "grade3", "3级高血压", "danger"
    if sys_v >= 160 or dia_v >= 100:
        return "grade2", "2级高血压", "high"
    if sys_v >= 140 or dia_v >= 90:
        return "grade1", "1级高血压", "high"
    if sys_v >= 120 or dia_v >= 80:
        return "high_normal", "正常高值", "attention"
    if sys_v < 90 or dia_v < 60:
        return "low", "血压偏低", "low"
    return "normal", "正常血压", "normal"


def evaluate(entry):
    if not isinstance(entry, dict):
        return "normal", ""
    kind = entry.get("kind")

    if kind == "bp":
        sys_v, dia_v = entry.get("sys") or 0, entry.get("dia") or 0
        grade, name, level = bp_grade(sys_v, dia_v)
        entry["grade"] = grade
        entry["grade_name"] = name
        phrase = {
            "danger": "vital_bp_danger",
            "high": "vital_bp_high",
            "attention": "vital_bp_high",
            "low": "vital_bp_low",
        }.get(level, "vital_ok")
        return level, phrase

    if kind == "glucose":
        v = entry.get("value") or 0
        if v >= 11.1 or v < 3.0:
            return "danger", "vital_glucose_danger"
        if v >= 7.0:
            return "high", "vital_glucose_high"
        if v < 3.9:
            return "low", "vital_glucose_low"
        return "normal", "vital_ok"

    if kind == "pulse":
        v = entry.get("value") or 0
        if v >= 130 or v < 40:
            return "danger", "vital_pulse_danger"
        if v > 100 or v < 55:
            return "high", "vital_pulse_high"
        return "normal", "vital_ok"

    if kind == "spo2":
        v = entry.get("value") or 0
        if v < 90:
            return "danger", "vital_spo2_danger"
        if v < 94:
            return "low", "vital_spo2_low"
        return "normal", "vital_ok"

    if kind == "temp":
        v = entry.get("value") or 0
        if v >= 39.0:
            return "danger", "vital_temp_danger"
        if v >= 37.3:
            return "high", "vital_temp_high"
        return "normal", "vital_ok"

    return "normal", "vital_ok"


def record(entry, level="normal", note=""):
    os.makedirs(HOME_DIR, exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    rec.update(entry or {})
    rec["level"] = level
    if note:
        rec["note"] = note
    try:
        with open(VITALS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass
    return rec


def load(limit=500):
    os.makedirs(HOME_DIR, exist_ok=True)
    out = []
    try:
        with open(VITALS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return out[-limit:]


def summary(limit=8):
    items = load()
    recent = items[-limit:][::-1]
    by_kind = {}
    for rec in items:
        by_kind.setdefault(rec.get("kind") or "?", []).append(rec)
    stats = {}
    for kind, rows in by_kind.items():
        vals = [r.get("value") for r in rows if isinstance(r.get("value"), (int, float))]
        if kind == "bp":
            sys_vals = [r.get("sys") for r in rows if isinstance(r.get("sys"), (int, float))]
            if sys_vals:
                stats["bp"] = {"count": len(rows), "avg_sys": round(sum(sys_vals) / len(sys_vals)),
                               "max_sys": max(sys_vals), "min_sys": min(sys_vals)}
        elif vals:
            stats[kind] = {"count": len(rows), "avg": round(sum(vals) / len(vals), 1),
                           "max": max(vals), "min": min(vals)}
    return {"recent": recent, "stats": stats, "total": len(items)}


ADVICE_TEXT = {
    "vital_ok": "好的，这次数值正常，已经记到健康档案里了。",
    "vital_bp_high": "这次血压偏高，安静休息十分钟再复测一次，还是高就联系医生。",
    "vital_bp_danger": "血压明显偏高，如果伴着头痛胸闷，请马上打幺二零。",
    "vital_bp_low": "血压偏低，先坐下或者躺下，起来的时候慢一点。",
    "vital_glucose_high": "血糖偏高，注意饮食，按医生说的按时吃药。",
    "vital_glucose_low": "血糖偏低，赶紧吃点糖或者点心，别空着肚子。",
    "vital_glucose_danger": "血糖很不正常，请马上联系医生或者打幺二零。",
    "vital_pulse_high": "心跳偏快，先安静休息一会儿再测一次。",
    "vital_pulse_danger": "心跳明显不正常，请马上联系医生。",
    "vital_spo2_low": "血氧偏低，如果觉得喘不上气，请马上打幺二零。",
    "vital_spo2_danger": "血氧很低，请马上打幺二零。",
    "vital_temp_high": "有点发烧，多喝水多休息，观察体温变化。",
    "vital_temp_danger": "体温比较高，请尽快联系医生或者去医院。",
}
