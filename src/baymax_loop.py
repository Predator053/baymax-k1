#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional


class Level(IntEnum):

    NORMAL = 0
    ATTENTION = 1
    WARNING = 2
    CRITICAL = 3

    @property
    def label(self) -> str:
        return {
            Level.NORMAL: "正常",
            Level.ATTENTION: "关注",
            Level.WARNING: "警告",
            Level.CRITICAL: "危急",
        }[self]


class Action:
    SPEAK = "speak"
    ASK = "ask"
    CAPTURE = "capture"
    DINGTALK = "dingtalk"
    SMS = "sms"
    CALL = "call"
    RECORD = "record"
    ESCALATE = "escalate"


@dataclass
class Decision:
    level: Level
    reason: str
    speak_key: Optional[str] = None
    actions: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def trace(self) -> Dict[str, Any]:
        return {
            "level": int(self.level),
            "level_label": self.level.label,
            "reason": self.reason,
            "speak": self.speak_key,
            "actions": self.actions,
            "data": self.data,
        }


OKAY_WORDS = ("没事", "没事儿", "不用", "别通知", "不要通知", "我挺好", "还好",
              "没摔着", "没受伤", "不用叫", "别叫", "我很好")

SOS_WORDS = ("救命", "快来人", "打120", "叫救护车", "我不行了", "喘不上气",
             "胸口疼", "胸口痛", "心口疼", "动不了", "起不来", "流血",
             "晕过去", "抽过去", "说不出话")

VITAL_RULES = [
    ("bp_sys", lambda v: v >= 180, Level.CRITICAL, "收缩压≥180", "vital_vital_bp_danger"),
    ("bp_sys", lambda v: 140 <= v < 180, Level.ATTENTION, "收缩压偏高", "vital_vital_bp_high"),
    ("bp_sys", lambda v: 0 < v < 90, Level.WARNING, "收缩压低", "vital_vital_bp_low"),
    ("glucose", lambda v: v >= 16.7, Level.CRITICAL, "血糖≥16.7", "vital_vital_glucose_danger"),
    ("glucose", lambda v: 7.0 <= v < 16.7, Level.ATTENTION, "血糖偏高", "vital_vital_glucose_high"),
    ("glucose", lambda v: 0 < v < 3.9, Level.WARNING, "血糖偏低", "vital_vital_glucose_low"),
    ("pulse", lambda v: v >= 130, Level.CRITICAL, "心率≥130", "vital_vital_pulse_danger"),
    ("pulse", lambda v: 100 <= v < 130, Level.ATTENTION, "心率偏快", "vital_vital_pulse_high"),
    ("spo2", lambda v: 0 < v <= 90, Level.CRITICAL, "血氧≤90", "vital_vital_spo2_danger"),
    ("spo2", lambda v: 90 < v <= 94, Level.WARNING, "血氧偏低", "vital_vital_spo2_low"),
    ("temp", lambda v: v >= 39.0, Level.CRITICAL, "体温≥39", "vital_vital_temp_danger"),
    ("temp", lambda v: 37.3 <= v < 39.0, Level.ATTENTION, "发热", "vital_vital_temp_high"),
]


def _said_okay(reply: str) -> bool:
    r = (reply or "").strip()
    return bool(r) and any(w in r for w in OKAY_WORDS)


def _said_sos(reply: str) -> bool:
    r = (reply or "").strip()
    return bool(r) and any(w in r for w in SOS_WORDS)


def assess_fall(g_force: float = 0.0, reply: str = "",
                faces: Optional[int] = None,
                position: Optional[str] = None) -> Decision:
    data = {"g": round(g_force, 2), "reply": reply, "faces": faces, "position": position}
    if _said_okay(reply):
        return Decision(
            level=Level.NORMAL,
            reason="用户自述无碍，闭环在本地终止（不惊动家属）",
            speak_key="fall_cancel",
            actions=[Action.SPEAK, Action.RECORD],
            data=data,
        )
    if _said_sos(reply):
        return Decision(
            level=Level.CRITICAL,
            reason="用户明确呼救",
            speak_key="fall_no_answer",
            actions=[Action.SPEAK, Action.CAPTURE, Action.DINGTALK, Action.SMS,
                     Action.CALL, Action.RECORD],
            data=data,
        )
    if not reply:
        return Decision(
            level=Level.CRITICAL,
            reason="跌倒后无人应答（6 秒无语音）",
            speak_key="fall_no_answer",
            actions=[Action.SPEAK, Action.CAPTURE, Action.DINGTALK, Action.SMS,
                     Action.CALL, Action.RECORD],
            data=data,
        )
    return Decision(
        level=Level.WARNING,
        reason="跌倒后有回应但未确认无碍",
        speak_key="fall_confirm",
        actions=[Action.SPEAK, Action.CAPTURE, Action.DINGTALK, Action.SMS, Action.RECORD],
        data=data,
    )


def assess_vitals(vitals: Dict[str, Any]) -> Decision:
    worst_level = Level.NORMAL
    best_key = None
    reasons = []
    for field_name, rule, level, why, key in VITAL_RULES:
        value = vitals.get(field_name)
        if value is None:
            continue
        try:
            if rule(float(value)):
                reasons.append(why)
                if level > worst_level:
                    worst_level = level
                    best_key = key
        except (TypeError, ValueError):
            continue
    if best_key is None:
        return Decision(
            level=Level.NORMAL,
            reason="数值在正常范围",
            speak_key="vital_vital_ok",
            actions=[Action.SPEAK, Action.RECORD],
            data={"vitals": vitals},
        )
    actions = [Action.SPEAK, Action.RECORD]
    if worst_level >= Level.WARNING:
        actions.append(Action.DINGTALK)
    return Decision(
        level=worst_level,
        reason="；".join(reasons),
        speak_key=best_key,
        actions=actions,
        data={"vitals": vitals},
    )


def assess_utterance(text: str) -> Decision:
    if not text:
        return Decision(level=Level.NORMAL, reason="空输入", actions=[])
    if _said_sos(text):
        return Decision(
            level=Level.CRITICAL,
            reason="语音中包含危急症状/求助",
            speak_key="emergency_ask",
            actions=[Action.SPEAK, Action.DINGTALK, Action.SMS, Action.RECORD],
            data={"text": text},
        )
    return Decision(level=Level.NORMAL, reason="普通对话", actions=[Action.RECORD],
                    data={"text": text})


class PendingAlert:

    def __init__(self, decision: Decision, escalate_after: float = 45.0):
        self.decision = decision
        self.created = time.time()
        self.escalate_after = escalate_after
        self.cancelled = False
        self.escalated = False
        self.outcome = "pending"

    def cancel(self, why: str = "user_okay") -> None:
        self.cancelled = True
        self.outcome = why

    def should_escalate(self) -> bool:
        return (
            not self.cancelled
            and not self.escalated
            and (time.time() - self.created) >= self.escalate_after
        )

    def mark_escalated(self) -> None:
        self.escalated = True
        self.outcome = "escalated_to_call"

    def summary(self) -> Dict[str, Any]:
        return {
            "level": int(self.decision.level),
            "reason": self.decision.reason,
            "outcome": self.outcome,
            "cancelled": self.cancelled,
            "escalated": self.escalated,
            "elapsed": round(time.time() - self.created, 1),
        }


class ClosedLoop:

    def __init__(self):
        self.pending: Optional[PendingAlert] = None
        self.history: List[Dict[str, Any]] = []

    def open(self, decision: Decision, escalate_after: float = 45.0) -> PendingAlert:
        alert = PendingAlert(decision, escalate_after=escalate_after)
        self.pending = alert
        self.history.append({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "opened": decision.trace(),
        })
        return alert

    def feedback(self, text: str) -> Optional[str]:
        if self.pending is None:
            return None
        if _said_okay(text):
            self.pending.cancel("user_okay")
            self.history.append({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "feedback": {"text": text, "action": "cancel"},
            })
            return "cancelled"
        if _said_sos(text):
            self.pending.decision.level = Level.CRITICAL
            self.history.append({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "feedback": {"text": text, "action": "raise_to_critical"},
            })
            return "escalate"
        return None

    def close(self, outcome: str = "closed") -> None:
        if self.pending is not None:
            self.history.append({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "closed": self.pending.summary(),
            })
            self.pending.outcome = outcome
            self.pending = None


DISCLAIMER_SHORT = "大白是健康辅助工具，不能代替医生诊断。"
DISCLAIMER_FULL = (
    "需要说明的是：我可以帮您了解健康知识、记录身体数据、在紧急时通知家人，"
    "但我不是医生，不能代替医生诊断，也不能代替您拨打急救电话。"
)
