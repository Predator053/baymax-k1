#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.getenv("BAYMAX_CONFIG", os.path.join(BASE_DIR, "config.json"))

try:
    import numpy as np
except Exception:
    np = None


def _float_env(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_CFG = _load_config()
_SEN_CFG = (_CFG.get("sensors") or {})
_THR = _CFG.get("thresholds") or {}

SIM_MODE = str(os.getenv("BAYMAX_HW_SIM", "auto")).strip().lower()

_MPU_CFG = (_SEN_CFG.get("mpu6050") or {})
MPU_BUS = int(os.getenv("BAYMAX_MPU_BUS", "") or _MPU_CFG.get("bus") or 0)
MPU_ADDRS = (0x68, 0x69)
FALL_FREEFALL_G = _float_env("BAYMAX_FALL_FREEFALL_G", _THR.get("fall_freefall_g", 0.50))
FALL_IMPACT_G = _float_env("BAYMAX_FALL_IMPACT_G", _THR.get("fall_impact_g", 2.20))
FALL_STILL_SEC = _float_env("BAYMAX_FALL_STILL_SEC", _THR.get("fall_still_sec", 1.2))
FALL_STILL_TOL = _float_env("BAYMAX_FALL_STILL_TOL", _THR.get("fall_still_tol", 0.30))
FALL_COOLDOWN = _float_env("BAYMAX_FALL_COOLDOWN", _THR.get("fall_cooldown", 60.0))

MAX30102_ADDR = 0x57
MAX30102_FS = 100.0

EC20_SMS_PORT = (os.getenv("BAYMAX_EC20_PORT") or os.getenv("EC20_SMS_PORT") or "").strip()
ALERT_PHONE = (os.getenv("ALERT_PHONE") or "").strip()

CAMERA_INDEX = int(os.getenv("BAYMAX_CAM_INDEX", "0") or 0)

_SKIP_BUS = set()
for _x in str(os.getenv("BAYMAX_I2C_SKIP", "9")).split(","):
    _x = _x.strip()
    if _x.isdigit():
        _SKIP_BUS.add(int(_x))
_BUS_ORDER = [1, 0, 2, 3, 4, 5, 6, 7, 8]


def _import_smbus():
    try:
        import smbus2 as m
        return m
    except ImportError:
        pass
    try:
        import smbus as m
        return m
    except ImportError:
        return None


SMBUS = _import_smbus()


def _bus_candidates(preferred=0):
    order = []
    if preferred:
        order.append(int(preferred))
    for b in _BUS_ORDER:
        if b not in order:
            order.append(b)
    return [b for b in order if b not in _SKIP_BUS and os.path.exists("/dev/i2c-%d" % b)]


def _probe_addr(bus_no, addr):
    if SMBUS is None:
        return None
    bus = None
    try:
        bus = SMBUS.SMBus(bus_no)
        bus.read_byte(addr)
        return bus
    except Exception:
        if bus is not None:
            try:
                bus.close()
            except Exception:
                pass
        return None


def find_i2c(addr, preferred=0, scan_timeout=6.0):
    if SMBUS is None:
        return None, None
    result = {}

    def _scan():
        for b in _bus_candidates(preferred):
            bus = _probe_addr(b, addr)
            if bus is not None:
                result["hit"] = (b, bus)
                return
        result["hit"] = (None, None)

    th = threading.Thread(target=_scan, name="i2c-scan", daemon=True)
    th.start()
    th.join(scan_timeout)
    return result.get("hit", (None, None))


MPU_REG_SMPLRT_DIV = 0x19
MPU_REG_CONFIG = 0x1A
MPU_REG_ACCEL_CONFIG = 0x1C
MPU_REG_ACCEL_XOUT_H = 0x3B
MPU_REG_PWR_MGMT_1 = 0x6B
MPU_REG_WHO_AM_I = 0x75

_MPU_WHOAMI_OK = {0x68, 0x69, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x98}
_ACCEL_LSB_PER_G = 8192.0


class Mpu6050:

    def __init__(self, bus=None, addr=None, preferred_bus=MPU_BUS):
        self.bus_no, self.bus = None, None
        candidates = [addr] if addr else list(MPU_ADDRS)
        kernel_err = None
        if SMBUS is not None:
            try:
                self._open_kernel(bus, candidates, preferred_bus)
            except Exception as e:
                kernel_err = e
        if self.bus is None:
            self._open_soft(candidates, kernel_err)
        self.who_am_i = getattr(self, "who_am_i", None) or self.bus.read_byte_data(
            self.addr, MPU_REG_WHO_AM_I)
        self._init_chip()
        self.offset = [0.0, 0.0, 0.0]
        self.gain = 1.0

    def _open_kernel(self, bus, candidates, preferred_bus):
        if bus is not None:
            self.bus = SMBUS.SMBus(bus)
            self.bus_no = bus
            for a in candidates:
                try:
                    if self.bus.read_byte_data(a, MPU_REG_WHO_AM_I) in _MPU_WHOAMI_OK:
                        self.addr = a
                        return
                except Exception:
                    continue
            self.bus.close()
            self.bus = None
            raise OSError("总线 %s 上没有 %s" % (bus, "/".join(hex(c) for c in candidates)))

        last = None
        for a in candidates:
            try:
                bus_no, handle = find_i2c(a, preferred_bus)
            except Exception as e:
                last = e
                continue
            if handle is None:
                last = a
                continue
            try:
                who = handle.read_byte_data(a, MPU_REG_WHO_AM_I)
            except Exception as e:
                last = e
                handle.close()
                continue
            if who in _MPU_WHOAMI_OK:
                self.bus, self.bus_no, self.addr = handle, bus_no, a
                self.who_am_i = who
                return
            handle.close()
        buses = _bus_candidates(preferred_bus)
        raise OSError("内核 I2C 没找到 MPU6050（已扫 i2c-%s）" % "/".join(str(b) for b in buses))

    def _open_soft(self, candidates, kernel_err):
        try:
            import soft_i2c
        except Exception as e:
            raise OSError("没找到 MPU6050（内核 I2C 不可用：%s；软件 I2C 模块缺失：%s）"
                          % (kernel_err, e))
        s = None
        try:
            s = soft_i2c.SoftI2C()
            for a in candidates:
                if not s.probe(a):
                    continue
                who = s.read(a, MPU_REG_WHO_AM_I, 1)[0]
                if who in _MPU_WHOAMI_OK:
                    self.bus, self.bus_no, self.addr = s, "soft", a
                    self.who_am_i = who
                    return
        except Exception as e:
            if s is not None:
                s.close()
            raise OSError("没找到 MPU6050（内核 I2C：%s；软件 I2C：%s）" % (kernel_err, e))
        if s is not None:
            s.close()
        raise OSError("没找到 MPU6050（内核 I2C：%s；软件 I2C 无应答，检查 3/5 针接线）"
                      % kernel_err)

    def _init_chip(self):
        b, a = self.bus, self.addr
        b.write_byte_data(a, MPU_REG_PWR_MGMT_1, 0x80)
        time.sleep(0.12)
        b.write_byte_data(a, MPU_REG_PWR_MGMT_1, 0x01)
        b.write_byte_data(a, MPU_REG_SMPLRT_DIV, 0x04)
        b.write_byte_data(a, MPU_REG_CONFIG, 0x03)
        b.write_byte_data(a, MPU_REG_ACCEL_CONFIG, 0x08)
        time.sleep(0.05)

    @staticmethod
    def _s16(hi, lo):
        v = (hi << 8) | lo
        return v - 65536 if v >= 32768 else v

    def read_raw_g(self):
        raw = self.bus.read_i2c_block_data(self.addr, MPU_REG_ACCEL_XOUT_H, 6)
        return (self._s16(raw[0], raw[1]) / _ACCEL_LSB_PER_G,
                self._s16(raw[2], raw[3]) / _ACCEL_LSB_PER_G,
                self._s16(raw[4], raw[5]) / _ACCEL_LSB_PER_G)

    def read_g(self):
        ax, ay, az = self.read_raw_g()
        return (ax - self.offset[0], ay - self.offset[1], az - self.offset[2])

    def read_magnitude(self):
        ax, ay, az = self.read_g()
        return math.sqrt(ax * ax + ay * ay + az * az) * self.gain

    def read_temperature(self):
        raw = self.bus.read_i2c_block_data(self.addr, 0x41, 2)
        return self._s16(raw[0], raw[1]) / 340.0 + 36.53

    def calibrate(self, samples=40, period=0.02):
        accs = []
        for _ in range(max(10, int(samples))):
            try:
                accs.append(self.read_raw_g())
            except Exception:
                pass
            time.sleep(period)
        if not accs:
            return False
        n = float(len(accs))
        mags = [math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2) for a in accs]
        mean_mag = sum(mags) / n
        if 0.5 < mean_mag < 2.0:
            self.gain = min(1.15, max(0.85, 1.0 / mean_mag))
        return True

    def posture(self, ax=None, ay=None, az=None):
        if ax is None:
            ax, ay, az = self.read_g()
        mag = math.sqrt(ax * ax + ay * ay + az * az)
        if mag < 0.2:
            return 0.0
        return max(0.0, min(1.0, 1.0 - abs(az) / mag))

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass

    def __str__(self):
        bus = "软件I2C" if self.bus_no == "soft" else "总线%s" % self.bus_no
        return "%s 地址0x%02X WHO_AM_I=0x%02X" % (bus, self.addr, self.who_am_i)


class FallStateMachine:

    def __init__(self, freefall_g=FALL_FREEFALL_G, impact_g=FALL_IMPACT_G,
                 still_sec=FALL_STILL_SEC, still_tol=FALL_STILL_TOL,
                 merge_window=1.5, cooldown=FALL_COOLDOWN):
        self.freefall_g = freefall_g
        self.impact_g = impact_g
        self.still_sec = still_sec
        self.still_tol = still_tol
        self.merge_window = merge_window
        self.cooldown = cooldown
        self.reset()

    def reset(self):
        self.state = "idle"
        self.t_state = 0.0
        self.t_still = None
        self.peak = 0.0
        self.last_fall = -1e9

    def update(self, now, g, horizontal=None):
        if now - self.last_fall < self.cooldown:
            return False
        quiet = abs(g - 1.0) <= self.still_tol

        if g < self.freefall_g:
            if self.state != "freefall":
                self.state, self.t_state, self.t_still = "freefall", now, None
                self.peak = g
        elif g > self.impact_g:
            if self.state == "freefall" and (now - self.t_state) > self.merge_window:
                self.state = "idle"
            if self.state in ("freefall", "idle"):
                self.state, self.t_state, self.t_still = "impact", now, None
                self.peak = g
            else:
                self.peak = max(self.peak, g)
        else:
            if self.state == "impact":
                if quiet:
                    if self.t_still is None:
                        self.t_still = now
                    if (now - self.t_still) >= self.still_sec:
                        self.state = "idle"
                        self.last_fall = now
                        return True
                else:
                    self.t_still = None
                    if (now - self.t_state) > self.merge_window:
                        self.state = "idle"
            elif self.state == "freefall" and (now - self.t_state) > self.merge_window:
                self.state = "idle"

        if self.state != "idle" and (now - self.t_state) > 10.0:
            self.state = "idle"
        return False


_FALL_SCRIPT = [
    (0.00, 1.00), (0.10, 0.85), (0.18, 0.42), (0.24, 0.30), (0.30, 0.55),
    (0.36, 2.90), (0.42, 3.40), (0.48, 1.80), (0.55, 1.10), (0.62, 0.98),
    (0.80, 1.01), (1.00, 1.00),
]


class SimMpu6050:

    def __init__(self, bus=0, addr=0x68):
        self.bus_no, self.addr = bus, addr
        self.who_am_i = 0x68
        self._inject_at = None
        self._tz = time.monotonic()
        self.offset = (0.0, 0.0, 0.0)
        self.gain = 1.0
        self.enabled = True

    def inject_fall(self):
        self._inject_at = time.monotonic()

    def read_g(self):
        t = time.monotonic() - self._tz
        az = 1.0 + 0.004 * math.sin(t * 1.7) + 0.002 * math.sin(t * 7.3)
        ax = 0.004 * math.sin(t * 0.9)
        ay = 0.003 * math.cos(t * 1.3)
        if self._inject_at is not None:
            el = time.monotonic() - self._inject_at
            last = _FALL_SCRIPT[-1][0]
            if el > last + 0.5:
                self._inject_at = None
            else:
                g = self._script_g(el)
                mag = math.sqrt(1.0 + g * g)
                az = g * az / max(1e-6, mag) if mag else g
                az = g * 0.98
                ax += g * 0.06
        return ax, ay, az

    @staticmethod
    def _script_g(el):
        prev = _FALL_SCRIPT[0]
        for item in _FALL_SCRIPT:
            if el <= item[0]:
                t0, g0 = prev
                t1, g1 = item
                if t1 <= t0:
                    return g1
                k = (el - t0) / (t1 - t0)
                return g0 + (g1 - g0) * k
            prev = item
        return _FALL_SCRIPT[-1][1]

    def read_magnitude(self):
        ax, ay, az = self.read_g()
        return math.sqrt(ax * ax + ay * ay + az * az)

    def posture(self, ax=None, ay=None, az=None):
        return 0.75 if self._inject_at is not None else 0.05

    def read_temperature(self):
        return 36.5

    def calibrate(self, samples=40, period=0.02):
        return True

    def close(self):
        pass

    def __str__(self):
        return "模拟(未检测到硬件)"


class FallMonitor:

    def __init__(self, on_fall=None, hz=20.0, sensor=None, allow_sim=None):
        self.on_fall = on_fall
        self.hz = max(5.0, float(hz))
        self.period = 1.0 / self.hz
        self.machine = FallStateMachine()
        self._sensor = sensor
        self.allow_sim = allow_sim
        self._stop = threading.Event()
        self._thread = None
        self.error = None
        self.available = False
        self.simulated = False
        self.who = ""
        self.last_g = 1.0

    def _make_sensor(self):
        if self._sensor is not None:
            return self._sensor
        try:
            s = Mpu6050()
            pass
            return s
        except Exception as e:
            reason = "%s: %s" % (type(e).__name__, e)
            if _sim_allowed(self.allow_sim):
                self.simulated = True
                pass
                return SimMpu6050()
            raise OSError(reason)

    def start(self):
        try:
            self._sensor = self._make_sensor()
        except Exception as e:
            self.error = "%s: %s" % (type(e).__name__, e)
            return False
        self.simulated = isinstance(self._sensor, SimMpu6050)
        self.who = str(self._sensor)
        self.available = True
        self._thread = threading.Thread(target=self._run, name="FallMonitor", daemon=True)
        self._thread.start()
        return True

    def _run(self):
        next_at = time.monotonic()
        calibrated = False
        t_start = next_at
        while not self._stop.is_set():
            next_at += self.period
            try:
                ax, ay, az = self._sensor.read_g()
            except Exception as e:
                self.error = "%s: %s" % (type(e).__name__, e)
                self.available = False
                print("\n⚠️ [外设] MPU6050 读失败，跌倒检测停止：%s" % self.error)
                return
            g = math.sqrt(ax * ax + ay * ay + az * az)
            self.last_g = g
            now = time.monotonic()
            if not calibrated and now - t_start > 2.0 and not self.simulated:
                try:
                    self._sensor.calibrate(samples=30)
                except Exception:
                    pass
                calibrated = True
            horiz = 0.0
            try:
                horiz = self._sensor.posture(ax, ay, az)
            except Exception:
                pass
            if self.machine.update(now, g, horiz) and self.on_fall:
                try:
                    self.on_fall(max(self.machine.peak, g))
                except Exception as e:
                    print("\n⚠️ [跌倒] 回调异常：%s" % e)
            sleep = next_at - time.monotonic()
            if sleep > 0:
                self._stop.wait(sleep)
            else:
                next_at = time.monotonic()

    def inject_fall(self):
        fn = getattr(self._sensor, "inject_fall", None)
        if fn is None:
            return False
        try:
            self.machine.reset()
            self.machine.last_fall = -1e9
        except Exception:
            pass
        fn()
        return True

    def stop(self):
        self._stop.set()


MAX30102_REG_INTR_STATUS_1 = 0x00
MAX30102_REG_INTR_STATUS_2 = 0x01
MAX30102_REG_INTR_ENABLE_1 = 0x02
MAX30102_REG_INTR_ENABLE_2 = 0x03
MAX30102_REG_FIFO_WR_PTR = 0x04
MAX30102_REG_OVF_COUNTER = 0x05
MAX30102_REG_FIFO_RD_PTR = 0x06
MAX30102_REG_FIFO_DATA = 0x07
MAX30102_REG_FIFO_CONFIG = 0x08
MAX30102_REG_MODE_CONFIG = 0x09
MAX30102_REG_SPO2_CONFIG = 0x0A
MAX30102_REG_LED1_PA = 0x0C
MAX30102_REG_LED2_PA = 0x0D
MAX30102_REG_PART_ID = 0xFF
MAX30102_REG_RESET = 0xFE


class Max30102:

    def __init__(self, bus=None, preferred_bus=int(os.getenv("BAYMAX_MAX30102_BUS", "0") or 0),
                 led_pa=0x24):
        if SMBUS is None:
            raise OSError("没有 smbus2")
        if bus is not None:
            self.bus, self.bus_no = SMBUS.SMBus(bus), bus
        else:
            bus_no, handle = find_i2c(MAX30102_ADDR, preferred_bus)
            if handle is None:
                raise OSError("没找到 MAX30102（0x57），检查 SDA/SCL 和 3.3V 供电")
            self.bus, self.bus_no = handle, bus_no
        try:
            part = self.bus.read_byte_data(MAX30102_ADDR, MAX30102_REG_PART_ID)
        except Exception as e:
            raise OSError("读 PART_ID 失败：%s" % e)
        if part != 0x15:
            raise OSError("PART_ID=0x%02X，不是 MAX30102" % part)
        self.part_id = part
        self.led_pa = led_pa
        self._init_chip()

    def _init_chip(self):
        b = self.bus
        a = MAX30102_ADDR
        b.write_byte_data(a, MAX30102_REG_RESET, 0x40)
        for _ in range(20):
            time.sleep(0.01)
            try:
                if b.read_byte_data(a, MAX30102_REG_RESET) == 0:
                    break
            except Exception:
                pass
        b.read_byte_data(a, MAX30102_REG_INTR_STATUS_1)
        b.read_byte_data(a, MAX30102_REG_INTR_STATUS_2)
        b.write_byte_data(a, MAX30102_REG_FIFO_CONFIG, 0x4F)
        b.write_byte_data(a, MAX30102_REG_MODE_CONFIG, 0x03)
        b.write_byte_data(a, MAX30102_REG_SPO2_CONFIG, 0x27)
        b.write_byte_data(a, MAX30102_REG_LED1_PA, self.led_pa)
        b.write_byte_data(a, MAX30102_REG_LED2_PA, self.led_pa)
        b.write_byte_data(a, MAX30102_REG_FIFO_WR_PTR, 0x00)
        b.write_byte_data(a, MAX30102_REG_OVF_COUNTER, 0x00)
        b.write_byte_data(a, MAX30102_REG_FIFO_RD_PTR, 0x00)

    def read_fifo(self):
        b, a = self.bus, MAX30102_ADDR
        wr = b.read_byte_data(a, MAX30102_REG_FIFO_WR_PTR) & 0x1F
        rd = b.read_byte_data(a, MAX30102_REG_FIFO_RD_PTR) & 0x1F
        n = (wr - rd) & 0x1F
        if n <= 0:
            return []
        raw = b.read_i2c_block_data(a, MAX30102_REG_FIFO_DATA, n * 6)
        out = []
        for i in range(n):
            o = i * 6
            red = ((raw[o] << 16) | (raw[o + 1] << 8) | raw[o + 2]) & 0x03FFFF
            ir = ((raw[o + 3] << 16) | (raw[o + 4] << 8) | raw[o + 5]) & 0x03FFFF
            out.append((red, ir))
        return out

    def read_window(self, seconds=8.0, fs=MAX30102_FS):
        n_target = int(seconds * fs)
        red, ir = [], []
        deadline = time.monotonic() + seconds * 1.6 + 2.0
        while len(red) < n_target and time.monotonic() < deadline:
            try:
                batch = self.read_fifo()
            except Exception:
                break
            if not batch:
                time.sleep(0.02)
                continue
            for r, i in batch:
                red.append(r)
                ir.append(i)
            time.sleep(0.01)
        return red, ir

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass

    def __str__(self):
        return "总线%d 地址0x%02X PART_ID=0x%02X" % (self.bus_no, MAX30102_ADDR, self.part_id)


def _movavg(x, win):
    n = len(x)
    win = int(max(1, win))
    if win <= 1 or n == 0:
        return np.array(x, dtype=float)
    k = np.ones(win) / win
    pad = win // 2
    xp = np.pad(np.array(x, dtype=float), (pad, win - 1 - pad), mode="edge")
    return np.convolve(xp, k, mode="valid")[:n]


def _autocorr_period(y, fs, min_bpm=32.0, max_bpm=200.0):
    n = len(y)
    s = np.array(y, dtype=float)
    s = s - s.mean()
    if n < int(fs * 3) or float(np.dot(s, s)) <= 0:
        return 0, 0.0
    lag_min = int(fs * 60.0 / max_bpm)
    lag_max = min(int(fs * 60.0 / min_bpm), n - 2)
    if lag_max <= lag_min + 3:
        return 0, 0.0
    lags = list(range(lag_min - 1, lag_max + 2))
    corr = {}
    for lag in lags:
        if lag <= 0 or lag >= n - 1:
            continue
        a = s[:n - lag]
        b = s[lag:]
        denom = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
        if denom > 0:
            corr[lag] = float(np.dot(a, b)) / denom
    if not corr:
        return 0, 0.0
    best_val = max(corr.values())
    best_lag = max(corr, key=lambda k: corr[k])
    for lag in sorted(corr):
        if lag < lag_min:
            continue
        if corr[lag] >= best_val * 0.97 \
                and corr.get(lag - 1, -1) <= corr[lag] \
                and corr.get(lag + 1, -2) <= corr[lag]:
            best_lag = lag
            break
    y0 = corr.get(best_lag - 1, 0.0)
    y1 = corr.get(best_lag, 0.0)
    y2 = corr.get(best_lag + 1, 0.0)
    den = (y0 - 2 * y1 + y2)
    if abs(den) > 1e-9:
        shift = 0.5 * (y0 - y2) / den
        if -0.5 < shift < 0.5:
            best_lag = best_lag + shift
    return float(best_lag), float(max(0.0, best_val))


def analyze_ppg(red, ir, fs=MAX30102_FS, min_ir_dc=8000.0):
    if np is None or red is None or ir is None or len(red) == 0 or len(ir) == 0:
        return {"finger": False, "hr": 0, "spo2": 0, "quality": 0.0,
                "dc_ir": 0.0, "ac_ir": 0.0, "reason": "no-numpy-or-data"}
    r = np.array(red, dtype=float)
    i = np.array(ir, dtype=float)
    n = min(len(r), len(i))
    r, i = r[-n:], i[-n:]
    if n < int(fs * 3):
        return {"finger": False, "hr": 0, "spo2": 0, "quality": 0.0,
                "dc_ir": float(i.mean()), "ac_ir": 0.0, "reason": "too-short"}

    dc_ir = float(np.median(i))
    dc_red = float(np.median(r))
    if dc_ir < min_ir_dc:
        return {"finger": False, "hr": 0, "spo2": 0, "quality": 0.0,
                "dc_ir": dc_ir, "ac_ir": 0.0, "reason": "no-finger"}

    win = int(fs * 2.0)
    ir_ac = _movavg(i - _movavg(i, win), 5)
    red_ac = _movavg(r - _movavg(r, win), 5)
    ac_ir = float(np.std(ir_ac))
    ac_red = float(np.std(red_ac))
    if ac_ir <= 0 or dc_red <= 0 or dc_ir <= 0:
        return {"finger": True, "hr": 0, "spo2": 0, "quality": 0.0,
                "dc_ir": dc_ir, "ac_ir": ac_ir, "reason": "flat"}

    period, corr = _autocorr_period(ir_ac, fs)
    hr = int(round(60.0 * fs / period)) if period > 0 else 0
    if not (32 <= hr <= 200):
        hr = 0

    ratio = (ac_red / dc_red) / (ac_ir / dc_ir)
    spo2 = int(round(max(70.0, min(100.0, 110.0 - 25.0 * ratio))))

    pi = ac_ir / dc_ir
    quality = max(0.0, min(1.0, corr * min(1.0, pi / 0.004)))
    if hr == 0:
        quality *= 0.3
    return {"finger": True, "hr": hr, "spo2": spo2,
            "quality": round(quality, 2), "dc_ir": round(dc_ir, 1),
            "ac_ir": round(ac_ir, 1), "ratio": round(ratio, 3),
            "period_s": round(period / fs, 3) if period else 0.0,
            "reason": "ok"}


def synth_ppg(hr=72.0, spo2=98.0, seconds=8.0, fs=MAX30102_FS,
              dc_ir=60000.0, ac_ir_ratio=0.020, seed=1, noise=0.0004):
    if np is None:
        return [], []
    rng = np.random.RandomState(seed)
    t = np.arange(int(seconds * fs)) / fs
    ph = 2.0 * np.pi * hr / 60.0 * t
    pulse = (np.sin(ph) + 0.35 * np.sin(2 * ph + 0.6) + 0.12 * np.sin(3 * ph + 1.1))
    pulse = pulse / max(1e-6, np.max(np.abs(pulse)))
    R = (110.0 - float(spo2)) / 25.0
    dc_red = dc_ir * 0.82
    ir = dc_ir * (1.0 + ac_ir_ratio * pulse) + rng.normal(0, noise * dc_ir, t.size)
    red = dc_red * (1.0 + R * ac_ir_ratio * pulse) + rng.normal(0, noise * dc_red, t.size)
    return red, ir


class SimMax30102:

    def __init__(self, hr=72.0, spo2=98.0, bus=0):
        self.hr, self.spo2 = float(hr), float(spo2)
        self.bus_no = bus
        self.part_id = 0x15
        self._seed = 0

    def read_window(self, seconds=8.0, fs=MAX30102_FS):
        self._seed += 1
        return synth_ppg(hr=self.hr, spo2=self.spo2, seconds=seconds, fs=fs,
                         seed=self._seed)

    def close(self):
        pass

    def __str__(self):
        return "模拟(未检测到硬件)"


HEALTH_PHRASES = {
    "hr_ok": "心率正常，血氧也正常，这次数据记到档案里了。",
    "hr_fast": "心跳有点快，先坐下休息两分钟，再测一次。",
    "hr_slow": "心跳偏慢，如果觉得头晕没力气，请联系医生。",
    "spo2_low": "血氧偏低，请慢慢深呼吸，过一会儿再测一次。",
    "spo2_danger": "血氧明显偏低，请马上打幺二零。",
    "no_finger": "没检测到手指，请把手指放到传感器上再测一次。",
}


def phrase_for(hr, spo2, finger=True):
    if not finger:
        return "no_finger", HEALTH_PHRASES["no_finger"]
    if spo2 and spo2 < 90:
        return "spo2_danger", HEALTH_PHRASES["spo2_danger"]
    if spo2 and spo2 < 94:
        return "spo2_low", HEALTH_PHRASES["spo2_low"]
    if hr and hr > 100:
        return "hr_fast", HEALTH_PHRASES["hr_fast"]
    if hr and 0 < hr < 55:
        return "hr_slow", HEALTH_PHRASES["hr_slow"]
    return "hr_ok", HEALTH_PHRASES["hr_ok"]


class HealthMonitor:

    def __init__(self, on_result=None, sensor=None, seconds=8.0, allow_sim=None):
        self.on_result = on_result
        self.seconds = float(seconds)
        self._sensor = sensor
        self.allow_sim = allow_sim
        self._req = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self.error = None
        self.available = False
        self.simulated = False
        self.who = ""
        self.busy = False
        self.last = None

    def _make_sensor(self):
        if self._sensor is not None:
            return self._sensor
        try:
            s = Max30102()
            pass
            return s
        except Exception as e:
            reason = "%s: %s" % (type(e).__name__, e)
            if _sim_allowed(self.allow_sim):
                self.simulated = True
                pass
                return SimMax30102()
            raise OSError(reason)

    def start(self):
        try:
            self._sensor = self._make_sensor()
        except Exception as e:
            self.error = "%s: %s" % (type(e).__name__, e)
            return False
        self.simulated = isinstance(self._sensor, SimMax30102)
        self.who = str(self._sensor)
        self.available = True
        self._thread = threading.Thread(target=self._run, name="HealthMonitor", daemon=True)
        self._thread.start()
        return True

    def request(self):
        if not self.available:
            return False
        self._req.set()
        return True

    def _run(self):
        while not self._stop.is_set():
            if not self._req.wait(0.5):
                continue
            self._req.clear()
            if self.busy:
                continue
            self.busy = True
            try:
                red, ir = self._sensor.read_window(seconds=self.seconds)
                res = analyze_ppg(red, ir)
                res["simulated"] = self.simulated
                res["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self.last = res
                print("[外设] 心率血氧：hr=%s spo2=%s quality=%s finger=%s%s"
                      % (res.get("hr"), res.get("spo2"), res.get("quality"),
                         res.get("finger"), "（模拟）" if self.simulated else ""))
                if self.on_result:
                    try:
                        self.on_result(res)
                    except Exception as e:
                        print("⚠️ [外设] 心率回调异常：%s" % e)
            except Exception as e:
                self.error = "%s: %s" % (type(e).__name__, e)
                print("⚠️ [外设] 心率测量失败：%s" % self.error)
            finally:
                self.busy = False

    def stop(self):
        self._stop.set()
        self._req.set()


def net_online(host="223.5.5.5", port=53, timeout=2.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class MmcliModem:

    def __init__(self, phone=None, index=None):
        self.index = self._pick_index() if index is None else index
        self.phone = (phone or ALERT_PHONE or "").strip()
        self.port = "mmcli"
        self.error = None
        self.signal = None
        self.operator = ""

    @staticmethod
    def _pick_index():
        try:
            r = subprocess.run(["mmcli", "-L"], capture_output=True, text=True, timeout=8)
            out = (r.stdout or "") + (r.stderr or "")
            ids = re.findall(r"/Modem/(\d+)", out)
            if ids:
                return int(ids[0])
        except Exception:
            pass
        return 0

    def _run(self, *args, **kw):
        import subprocess
        return subprocess.run(["mmcli", "-m", str(self.index)] + list(args),
                              capture_output=True, text=True,
                              timeout=kw.get("timeout", 10))

    def open(self):
        import re
        try:
            r = self._run(timeout=8)
        except Exception as e:
            self.error = "%s: %s" % (type(e).__name__, e)
            return False
        out = (r.stdout or "") + (r.stderr or "")
        m = re.search(r"state:\s*(\S+)", out)
        state = m.group(1) if m else ""
        if r.returncode != 0 or not state:
            self.error = (out.strip().splitlines()[-1][:80] if out.strip() else "mmcli 不可用")
            return False
        m = re.search(r"signal quality:\s*(\d+)%", out)
        if m:
            self.signal = int(m.group(1))
        m = re.search(r"operator name:\s*(.+)", out)
        if m:
            self.operator = m.group(1).strip()
        if state not in ("registered", "connected", "enabled"):
            self.error = "模组状态 %s（未注册网络）" % state
            return False
        return True

    def send_sms(self, text, phone=None, timeout=25):
        import re
        phone = (phone or self.phone or "").strip()
        if not phone:
            self.error = "没配置告警号码 ALERT_PHONE"
            return False, self.error
        body = str(text).replace(",", "，").replace("'", "’")
        try:
            r = self._run("--messaging-create-sms=text='%s',number='%s'" % (body, phone),
                          timeout=15)
        except Exception as e:
            return False, "%s: %s" % (type(e).__name__, e)
        out = (r.stdout or "") + (r.stderr or "")
        m = re.search(r"SMS/(\d+)", out)
        if r.returncode != 0 or not m:
            tail = out.strip().splitlines()[-1][:70] if out.strip() else "无输出"
            return False, "创建短信失败：" + tail
        sid = m.group(1)
        try:
            r2 = self._run("-s", sid, "--send", timeout=timeout)
        except Exception as e:
            return False, "%s: %s" % (type(e).__name__, e)
        out2 = (r2.stdout or "") + (r2.stderr or "")
        if r2.returncode == 0 and "successfully" in out2.lower():
            return True, "已发送（4G/%s）" % (self.operator or "ModemManager")
        try:
            self._run("--messaging-delete-sms=%s" % sid, timeout=8)
        except Exception:
            pass
        tail = out2.strip().splitlines()[-1][:70] if out2.strip() else "无输出"
        return False, "发送失败：" + tail

    def close(self):
        pass

    def get_position(self):
        import re
        try:
            r = self._run("--location-get", timeout=6)
            out = (r.stdout or "") + (r.stderr or "")
            lat = re.search(r"latitude:\s*([-0-9.]+)", out)
            lon = re.search(r"longitude:\s*([-0-9.]+)", out)
            if lat and lon:
                la, lo = float(lat.group(1)), float(lon.group(1))
                if abs(la) > 0.001 and abs(lo) > 0.001:
                    return "%.4f,%.4f（卫星定位）" % (la, lo)
        except Exception:
            pass
        try:
            loc = (_CFG.get("location") or {}) if isinstance(_CFG, dict) else {}
            if loc.get("lat") and loc.get("lon"):
                return "%s %.4f,%.4f（预设位置）" % (
                    loc.get("name") or "安装位置",
                    float(loc["lat"]), float(loc["lon"]))
        except Exception:
            pass
        try:
            import json as _json
            import urllib.request
            url = ("http://ip-api.com/json/?lang=zh-CN"
                   "&fields=status,regionName,city,lat,lon")
            with urllib.request.urlopen(url, timeout=6) as f:
                d = _json.load(f)
            if d.get("status") == "success":
                where = "%s%s" % (d.get("regionName") or "", d.get("city") or "")
                return "%s %.4f,%.4f（网络定位）" % (
                    where or "未知地区", float(d.get("lat", 0)), float(d.get("lon", 0)))
        except Exception:
            pass
        return None

    def place_call(self, number=None, timeout=220):
        number = (number or self.phone or "").strip()
        if not number:
            return False, "没配置告警号码 ALERT_PHONE"
        helper = "/usr/local/bin/baymax_call.sh"
        if os.path.isfile(helper):
            last_out = ""
            for attempt in (1, 2):
                try:
                    r = subprocess.run(["sudo", "-n", helper, number, "20"],
                                       capture_output=True, text=True, timeout=timeout)
                except Exception as e:
                    return False, "%s: %s" % (type(e).__name__, e)
                out = (r.stdout or "") + (r.stderr or "")
                last_out = out
                if "ANSWERED" in out:
                    return True, "已接通 %s" % number
                if "RINGING" in out:
                    if attempt == 1:
                        print("[拨号] 第一通未接，90 秒后重拨")
                        time.sleep(90)
                        continue
                    return True, "两通均未接听 %s" % number
                tail = out.strip().splitlines()[-1][:70] if out.strip() else "无输出"
                return False, "拨号失败：" + tail
            tail = last_out.strip().splitlines()[-1][:70] if last_out.strip() else "无输出"
            return False, "拨号失败：" + tail

        try:
            r = self._run("--voice-create-call=number=%s" % number, timeout=15)
        except Exception as e:
            return False, "%s: %s" % (type(e).__name__, e)
        out = (r.stdout or "") + (r.stderr or "")
        if "Successfully created" in out:
            return True, "已拨号 %s" % number
        tail = out.strip().splitlines()[-1][:70] if out.strip() else "无输出"
        return False, tail

    def __str__(self):
        csq = (" CSQ=%s" % self.signal) if self.signal is not None else ""
        return "4G %s%s" % (self.operator or "", csq)


class Ec20:

    def __init__(self, port=None, phone=None, baud=115200):
        self.port = (port or EC20_SMS_PORT or "").strip()
        self.phone = (phone or ALERT_PHONE or "").strip()
        self.baud = baud
        self.fd = None
        self.error = None
        self.signal = None
        self.operator = ""

    def _open(self, port):
        import termios
        fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            attrs = termios.tcgetattr(fd)
            speed = getattr(termios, "B%d" % self.baud, termios.B115200)
            attrs[0] = 0
            attrs[1] = 0
            attrs[2] = termios.CLOCAL | termios.CREAD
            attrs[3] = 0
            attrs[4] = attrs[5] = speed
            attrs[6][termios.VMIN] = 0
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
            termios.tcflush(fd, termios.TCIOFLUSH)
        except Exception:
            pass
        return fd

    def open(self):
        if self.fd is not None:
            return True
        ports = []
        if self.port:
            ports.append(self.port)
        else:
            for pat in ("/dev/ttyUSB2", "/dev/ttyUSB3", "/dev/ttyUSB1",
                        "/dev/ttyUSB0", "/dev/ttyACM*"):
                ports.extend(sorted(glob.glob(pat)))
        for p in ports:
            if not os.path.exists(p):
                continue
            try:
                fd = self._open(p)
            except Exception as e:
                self.error = "%s: %s" % (type(e).__name__, e)
                continue
            self.fd = fd
            ok, _ = self.cmd("AT", timeout=1.2)
            if ok:
                self.port = p
                self.cmd("ATE0", timeout=1.0)
                self.cmd("AT+CMEE=1", timeout=1.0)
                return True
            self._close()
            self.error = "AT 无响应：%s" % p
        if not self.error:
            self.error = "没有可用的 AT 串口（EC20 未插入或驱动未加载）"
        return False

    def _close(self):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None

    def close(self):
        self._close()

    def _write(self, data):
        if isinstance(data, str):
            data = data.encode("ascii", "ignore")
        os.write(self.fd, data)

    def _read(self, seconds=1.0, stop_on=()):
        deadline = time.monotonic() + seconds
        buf = b""
        while time.monotonic() < deadline:
            try:
                chunk = os.read(self.fd, 512)
            except BlockingIOError:
                chunk = b""
            except OSError:
                break
            if chunk:
                buf += chunk
                deadline = max(deadline, time.monotonic() + 0.15)
                txt = buf.decode("utf-8", "ignore")
                if any(s in txt for s in stop_on):
                    break
            else:
                time.sleep(0.02)
        return buf.decode("utf-8", "ignore")

    def cmd(self, line, timeout=1.5, expect=("OK",)):
        if self.fd is None:
            return False, ""
        try:
            self._write(line + "\r")
        except OSError as e:
            self.error = "%s: %s" % (type(e).__name__, e)
            return False, ""
        resp = self._read(timeout)
        if "ERROR" in resp:
            return False, resp
        return (any(e in resp for e in expect) if expect else True), resp

    def status(self):
        info = {"port": self.port, "ready": False, "csq": None, "operator": ""}
        if not self.open():
            info["error"] = self.error
            return info
        ok, resp = self.cmd("AT+CPIN?", timeout=1.5, expect=("READY", "OK"))
        info["ready"] = "READY" in resp
        ok, resp = self.cmd("AT+CSQ", timeout=1.2, expect=("+CSQ",))
        try:
            val = resp.split("+CSQ:")[1].split(",")[0].strip()
            info["csq"] = int(val)
            self.signal = info["csq"]
        except Exception:
            pass
        ok, resp = self.cmd("AT+COPS?", timeout=2.5, expect=("+COPS",))
        if '"' in resp:
            info["operator"] = resp.split('"')[1]
            self.operator = info["operator"]
        return info

    @staticmethod
    def _ucs2(text):
        return "".join("%04X" % ord(c) for c in text if ord(c) <= 0xFFFF)

    def _sms_body(self, text, ucs2=True):
        if ucs2 and any(ord(c) > 127 for c in text):
            self.cmd('AT+CSCS="UCS2"', timeout=1.2)
            return self._ucs2(text), True
        self.cmd('AT+CSCS="GSM"', timeout=1.2)
        return "".join(c if 32 <= ord(c) < 127 else "?" for c in text), False

    def send_sms(self, text, phone=None, timeout=6.0):
        phone = (phone or self.phone or "").strip()
        if not phone:
            self.error = "没配置告警号码 ALERT_PHONE"
            return False, self.error
        if not self.open():
            return False, self.error or "串口打不开"
        ok, resp = self.cmd("AT+CMGF=1", timeout=1.5)
        if not ok:
            return False, "AT+CMGF=1 失败"
        for attempt in (True, False):
            body, is_ucs2 = self._sms_body(text, ucs2=attempt)
            num = self._ucs2(phone) if is_ucs2 else phone
            try:
                self._write('AT+CMGS="%s"\r' % num)
            except OSError as e:
                return False, "%s: %s" % (type(e).__name__, e)
            prompt = self._read(2.0, stop_on=(">",))
            if ">" not in prompt:
                continue
            try:
                self._write(body + "\x1a")
            except OSError as e:
                return False, "%s: %s" % (type(e).__name__, e)
            resp = self._read(timeout, stop_on=("+CMGS", "ERROR"))
            if "+CMGS" in resp or "OK" in resp:
                return True, "已发送（%s）" % ("UCS2" if is_ucs2 else "ASCII")
        return False, "模块拒绝了这条短信：%s" % (resp.strip()[:80] if resp else "无响应")

    def __str__(self):
        return "%s%s" % (self.port, (" CSQ=%s" % self.signal) if self.signal is not None else "")


class SimEc20:

    def __init__(self, port="sim", phone=None):
        self.port = "sim"
        self.phone = phone or ALERT_PHONE or "13800000000"
        self.signal = 20
        self.error = None
        home = os.getenv("BAYMAX_HOME") or os.path.expanduser("~/.baymax")
        self.outbox = os.path.join(home, "sms_outbox.log")

    def open(self):
        return True

    def close(self):
        pass

    def status(self):
        return {"port": "sim", "ready": True, "csq": 20, "operator": "模拟", "simulated": True}

    def send_sms(self, text, phone=None, timeout=0):
        try:
            os.makedirs(os.path.dirname(self.outbox), exist_ok=True)
            with open(self.outbox, "a", encoding="utf-8") as f:
                f.write("%s -> %s : %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                                             phone or self.phone, text))
        except Exception:
            pass
        return True, "模拟发送（已写入 sms_outbox.log）"

    def __str__(self):
        return "模拟(未检测到硬件)"


def send_sms(text, port=None, phone=None, timeout=3.0):
    mod = Ec20(port=port, phone=phone)
    try:
        ok, _ = mod.send_sms(text, phone=phone, timeout=timeout)
        return ok
    except Exception:
        return False
    finally:
        mod.close()


def send_alert(reason, detail=""):
    msg = "【大白告警】%s。%s" % (reason, detail) if detail else "【大白告警】%s" % reason
    hub = get_hub()
    last = "未知原因"
    for attempt in range(1, 4):
        try:
            modem = hub.get_modem()
        except Exception as e:
            modem = None
            last = "取模块失败：%s" % e
        if modem is not None:
            try:
                ok, info = modem.send_sms(msg)
                if ok:
                    return ok, info
                last = info
            except Exception as e:
                last = "%s: %s" % (type(e).__name__, e)
        if attempt < 3:
            time.sleep(3)
            hub.modem = None
    try:
        at = Ec20()
        if at.open():
            ok, info = at.send_sms(msg)
            if ok:
                return ok, info + "（AT 通道）"
    except Exception:
        pass
    return False, last


def get_position():
    try:
        modem = get_hub().get_modem()
    except Exception:
        return None
    fn = getattr(modem, "get_position", None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None


def place_call(number=None):
    try:
        modem = get_hub().get_modem()
    except Exception as e:
        return False, "没有 4G 模块（%s）" % e
    fn = getattr(modem, "place_call", None)
    if fn is None:
        return False, "当前短信后端不支持拨号"
    try:
        return fn(number)
    except Exception as e:
        return False, "%s: %s" % (type(e).__name__, e)


_CASCADE_PATHS = [
    "/opt/spacemit-ai-tool/cv2/data/haarcascade_frontalface_default.xml",
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
]


class _QuietStderr:

    def __enter__(self):
        try:
            self._old = os.dup(2)
            self._null = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self._null, 2)
        except OSError:
            self._old = None
        return self

    def __exit__(self, *exc):
        if getattr(self, "_old", None) is not None:
            try:
                os.dup2(self._old, 2)
                os.close(self._old)
                os.close(self._null)
            except OSError:
                pass
        return False


def _load_cascade():
    import cv2
    for p in _CASCADE_PATHS:
        if os.path.isfile(p):
            c = cv2.CascadeClassifier(p)
            if not c.empty():
                return c
    if hasattr(cv2, "data"):
        p = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.isfile(p):
            c = cv2.CascadeClassifier(p)
            if not c.empty():
                return c
    return None


class Camera:

    def __init__(self, index=None):
        self.index = CAMERA_INDEX if index is None else int(index)
        self.cap = None
        self.error = None
        self.size = ""

    def open(self, timeout=6.0):
        if self.cap is not None:
            return True
        import cv2

        box = {}

        def _do():
            try:
                with _QuietStderr():
                    cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
                    if not cap or not cap.isOpened():
                        if cap:
                            cap.release()
                        cap = cv2.VideoCapture(self.index)
                    if not cap or not cap.isOpened():
                        box["err"] = "打不开 /dev/video%d" % self.index
                        return
                    for _ in range(4):
                        cap.read()
                    box["cap"] = cap
            except Exception as e:
                box["err"] = "%s: %s" % (type(e).__name__, e)

        th = threading.Thread(target=_do, name="cam-open", daemon=True)
        th.start()
        th.join(timeout)
        if "cap" not in box:
            self.error = box.get("err") or "打开摄像头超时"
            return False
        self.cap = box["cap"]
        try:
            w = int(self.cap.get(3))
            h = int(self.cap.get(4))
            self.size = "%dx%d" % (w, h)
        except Exception:
            self.size = ""
        return True

    def grab(self):
        if not self.open():
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def count_faces(self, frame=None):
        try:
            import cv2
            if frame is None:
                frame = self.grab()
            if frame is None:
                return None
            cascade = _load_cascade()
            if cascade is None:
                self.error = "找不到 haarcascade 人脸模型"
                return None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.12, 5, minSize=(60, 60))
            return len(faces)
        except Exception as e:
            self.error = "%s: %s" % (type(e).__name__, e)
            return None

    def snapshot(self, path):
        frame = self.grab()
        if frame is None:
            return False
        try:
            import cv2
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            return bool(cv2.imwrite(path, frame))
        except Exception:
            return False

    def close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def __str__(self):
        return "/dev/video%d %s" % (self.index, self.size)


class SimCamera:

    def __init__(self, index=0, faces=1):
        self.index = index
        self.faces = faces
        self.size = "模拟"
        self.error = None

    def open(self, timeout=0):
        return True

    def grab(self):
        return None

    def count_faces(self, frame=None):
        return self.faces

    def snapshot(self, path):
        return False

    def close(self):
        pass

    def __str__(self):
        return "模拟(未检测到硬件)"


def camera_count_faces(index=None):
    cam = Camera(index)
    try:
        return cam.count_faces()
    finally:
        cam.close()


def _sim_allowed(explicit=None):
    if explicit is not None:
        return bool(explicit)
    if SIM_MODE in ("0", "off", "false", "no", "real"):
        return False
    return True


class Hub:

    def __init__(self):
        self.mpu = None
        self.max30 = None
        self.modem = None
        self.camera = None
        self.fall_monitor = None
        self.health_monitor = None
        self._lock = threading.Lock()
        self.info = {}

    def get_camera(self):
        if self.camera is not None:
            return self.camera
        try:
            cam = Camera()
            if cam.open():
                self.camera = cam
                return cam
            raise OSError(cam.error or "打开失败")
        except Exception as e:
            csi = csi_camera_status()
            if csi:
                self.info["camera"] = ("在线", csi)
                self.camera = SimCamera()
                return self.camera
            if _sim_allowed():
                self.camera = SimCamera()
                self.info["camera"] = ("模拟", "未检测到摄像头（%s）" % e)
                return self.camera
            self.info["camera"] = ("缺失", str(e))
            return None

    def get_modem(self):
        if isinstance(self.modem, SimEc20):
            if time.time() < getattr(self, "_modem_retry_at", 0):
                return self.modem
            self.modem = None
        if self.modem is not None:
            return self.modem
        try:
            mm = MmcliModem()
            if mm.open():
                self.modem = mm
                self.info["sms"] = ("在线", "4G %s 信号%s%%" %
                                    (mm.operator or "已注册", mm.signal))
                return mm
            self.info["sms"] = ("缺失", mm.error or "mmcli 不可用")
        except Exception as _e:
            self.info["sms"] = ("缺失", "mmcli: %s" % _e)

        try:
            m = Ec20()
            if m.open():
                self.modem = m
                return m
            raise OSError(m.error or "AT 无响应")
        except Exception as e:
            if _sim_allowed():
                self.modem = SimEc20()
                self._modem_retry_at = time.time() + 20.0
                self.info["sms"] = ("模拟", "未检测到 4G 模块（%s）" % e)
                return self.modem
            self.info["sms"] = ("缺失", str(e))
            return None

    def start_sensors(self, on_fall=None, on_result=None, fall_hz=20.0):
        fm = FallMonitor(on_fall=on_fall, hz=fall_hz)
        if fm.start():
            self.fall_monitor = fm
            self.mpu = fm._sensor
            self.info["mpu6050"] = ("模拟" if fm.simulated else "在线", fm.who)
        else:
            self.info["mpu6050"] = ("缺失", fm.error or "启动失败")
            print("[外设] MPU6050 未启用：%s" % (fm.error or "启动失败"))

        hm = HealthMonitor(on_result=on_result)
        if hm.start():
            self.health_monitor = hm
            self.max30 = hm._sensor
            self.info["max30102"] = ("模拟" if hm.simulated else "在线", hm.who)
        else:
            self.info["max30102"] = ("缺失", hm.error or "启动失败")
            print("[外设] MAX30102 未启用：%s" % (hm.error or "启动失败"))
        return self

    def status(self):
        rows = []

        def _add(dev, model, state, detail):
            rows.append({"dev": dev, "model": model, "state": state, "detail": detail})

        fm = self.fall_monitor
        if fm is None:
            _add("MPU6050", "六轴加速度", "缺失", "未启动")
        elif fm.available:
            _add("MPU6050", "六轴加速度", "模拟" if fm.simulated else "在线",
                 fm.who if not fm.simulated else "未检测到硬件，跌倒由模拟波形触发")
        else:
            _add("MPU6050", "六轴加速度", "缺失", fm.error or "读取失败")

        hm = self.health_monitor
        if hm is None:
            _add("MAX30102", "心率血氧", "缺失", "未启动")
        elif hm.available:
            _add("MAX30102", "心率血氧", "模拟" if hm.simulated else "在线",
                 hm.who if not hm.simulated else "未检测到硬件，用合成脉搏波")
        else:
            _add("MAX30102", "心率血氧", "缺失", hm.error or "启动失败")

        m = self.modem
        if m is None:
            _add("EC20", "4G短信", "未探测", "点面板「电话测试」或启动后自动探测")
        elif isinstance(m, SimEc20):
            _add("EC20", "4G短信", "模拟", "未检测到模块，短信写入 sms_outbox.log")
        else:
            _add("EC20", "4G短信", "在线", str(m))

        c = self.camera
        cam_info = self.info.get("camera") or ("", "")
        if c is None and not cam_info[0]:
            _add("摄像头", "UVC/CSI", "未探测", "需要时才打开，避免和麦克风抢 USB")
        elif cam_info[0] == "在线":
            _add("摄像头", "CSI", "在线", str(cam_info[1]))
        elif isinstance(c, SimCamera):
            _add("摄像头", "UVC/CSI", "模拟", "未检测到摄像头，返回固定人脸数")
        else:
            _add("摄像头", "UVC", "在线", str(c))

        _add("网络", "以太网/WiFi", "在线" if net_online() else "离线",
             "223.5.5.5:53 TCP 探测")
        return rows

    def write_status(self):
        home = os.getenv("BAYMAX_HOME") or os.path.expanduser("~/.baymax")
        data = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sim_mode": SIM_MODE,
            "devices": self.status(),
        }
        try:
            os.makedirs(home, exist_ok=True)
            path = os.path.join(home, "hw.json")
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception:
            pass
        return data

    def shutdown(self):
        for obj in (self.fall_monitor, self.health_monitor):
            try:
                if obj:
                    obj.stop()
            except Exception:
                pass
        for obj in (self.mpu, self.max30, self.modem, self.camera):
            try:
                if obj:
                    obj.close()
            except Exception:
                pass


_HUB = None
_HUB_LOCK = threading.Lock()


def get_hub():
    global _HUB
    with _HUB_LOCK:
        if _HUB is None:
            _HUB = Hub()
        return _HUB


def probe():
    info = {}
    try:
        if SMBUS is None:
            info["mpu6050"] = "不可用（缺 smbus2）"
        else:
            last = None
            for attempt in range(2):
                try:
                    s = Mpu6050()
                    info["mpu6050"] = "OK %s 当前%.2fg" % (s, s.read_magnitude())
                    s.close()
                    break
                except Exception as e:
                    last = e
                    time.sleep(0.6)
            else:
                raise last
    except Exception as e:
        info["mpu6050"] = "缺失（%s）" % e

    try:
        if SMBUS is None or np is None:
            info["max30102"] = "不可用（缺 smbus2/numpy）"
        else:
            s = Max30102()
            info["max30102"] = "OK %s" % s
            s.close()
    except Exception as e:
        info["max30102"] = "缺失（%s）" % e

    try:
        mm = MmcliModem()
        if mm.open():
            info["sms"] = "OK 4G %s 信号%s%%" % (mm.operator or "已注册", mm.signal)
        else:
            m = Ec20()
            if m.open():
                st = m.status()
                info["sms"] = "OK %s 信号%s %s" % (
                    st.get("port"), st.get("csq"), st.get("operator") or "")
                m.close()
            else:
                info["sms"] = "缺失（%s）" % (m.error or mm.error or "")
    except Exception as e:
        info["sms"] = "缺失（%s）" % e

    info["net"] = "在线" if net_online() else "离线"
    try:
        cam = Camera()
        faces = cam.count_faces() if cam.open() else None
        cam.close()
        if faces is not None:
            info["camera"] = "OK %s 人脸%d" % (cam.size, faces)
        else:
            info["camera"] = csi_camera_status() or "不可用"
    except Exception:
        info["camera"] = "不可用"
    return info


_CSI_CACHE = {"ts": 0.0, "val": None, "done": False}
_CSI_LOCK = threading.Lock()


def csi_camera_status(timeout=14, ttl=120):
    with _CSI_LOCK:
        now = time.time()
        if _CSI_CACHE["done"] and now - _CSI_CACHE["ts"] < ttl:
            return _CSI_CACHE["val"]
        val = _csi_camera_probe(timeout)
        _CSI_CACHE.update(ts=now, val=val, done=True)
        return val


def _csi_camera_probe(timeout=14):
    if not glob.glob("/dev/cam_sensor*"):
        return None
    cfg = "/usr/share/camera_json/csi3_camera_detect.json"
    if not os.path.isfile(cfg):
        return None
    env = dict(os.environ)
    env.setdefault("DISPLAY", ":0")
    try:
        r = subprocess.run(["/usr/bin/cam-test", cfg], capture_output=True,
                           text=True, timeout=timeout, env=env)
    except Exception:
        return None
    out = (r.stdout or "") + (r.stderr or "")
    if "detect sensor" in out and "success" in out:
        name = "ov5647" if "ov5647" in out.lower() else "CSI"
        fps = "30fps" if "fps: 30" in out else ""
        return ("OK CSI %s %s" % (name, fps)).strip()
    return None


_CSI_CFG = "/tmp/sdktest.json"
_CSI_AUTO = "/tmp/csi3_camera_auto.json"


def _ensure_csi_config():
    try:
        if not os.path.isfile(_CSI_AUTO):
            env = dict(os.environ)
            env.setdefault("DISPLAY", ":0")
            subprocess.run(["/usr/bin/cam-test",
                            "/usr/share/camera_json/csi3_camera_detect.json"],
                           capture_output=True, text=True, timeout=14, env=env)
        if os.path.isfile(_CSI_AUTO):
            with open(_CSI_AUTO, "rb") as src, open(_CSI_CFG, "wb") as dst:
                dst.write(src.read())
            return True
    except Exception:
        pass
    return os.path.isfile(_CSI_CFG)


def csi_snapshot(path, frames=12, width=1920, height=1080, timeout=45):
    if not _ensure_csi_config():
        return False
    import fcntl
    lock_path = "/tmp/baymax_camera.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    except OSError:
        lock_fd = None
    try:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            except OSError:
                pass
        ok = _csi_capture_once(path, frames, width, height, timeout)
        if not ok:
            time.sleep(1.5)
            ok = _csi_capture_once(path, frames, width, height, timeout)
        return ok
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            except OSError:
                pass


def _csi_capture_once(path, frames, width, height, timeout):
    raw = "/tmp/_baymax_frame.nv12"
    try:
        if os.path.exists(raw):
            os.remove(raw)
        r = subprocess.run(["gst-launch-1.0", "-q", "spacemitsrc",
                            "num-buffers=%d" % frames,
                            "!", "filesink", "location=%s" % raw],
                           capture_output=True, text=True, timeout=timeout)
        if not os.path.isfile(raw):
            return False
        with open(raw, "rb") as f:
            data = f.read()
        frame_size = width * height * 3 // 2
        n = len(data) // frame_size
        if n <= 0:
            return False
        blob = data[(n - 1) * frame_size: n * frame_size]
        arr = np.frombuffer(blob, dtype=np.uint8)
        y = arr[:width * height].reshape(height, width).astype(np.float32)
        uv = arr[width * height:].reshape(height // 2, width // 2, 2).astype(np.float32)
        u = np.repeat(np.repeat(uv[:, :, 0], 2, axis=0), 2, axis=1)
        v = np.repeat(np.repeat(uv[:, :, 1], 2, axis=0), 2, axis=1)
        c = y - 16.0
        d = u - 128.0
        e = v - 128.0
        rgb = np.stack([1.164 * c + 1.596 * e,
                        1.164 * c - 0.392 * d - 0.813 * e,
                        1.164 * c + 2.017 * d], axis=-1).clip(0, 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(rgb).save(path, quality=88)
        try:
            os.remove(raw)
        except OSError:
            pass
        return True
    except subprocess.TimeoutExpired:
        try:
            subprocess.run(["pkill", "-f", "gst-launch-1.0"],
                           capture_output=True, timeout=5)
        except Exception:
            pass
        print("[摄像头] 抓帧超时，已清理残留进程")
        return False
    except Exception as e:
        print("[摄像头] 抓帧失败：%s: %s" % (type(e).__name__, e))
        return False


def csi_face_count(image_path):
    try:
        import cv2
    except Exception:
        return None
    cascade = None
    for p in _CASCADE_PATHS:
        if os.path.isfile(p):
            cascade = cv2.CascadeClassifier(p)
            if not cascade.empty():
                break
    if cascade is None or cascade.empty():
        return None
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(gray, 1.15, 5, minSize=(60, 60))
        return int(len(faces))
    except Exception:
        return None


def hw_status():
    return get_hub().write_status()


def simulate_fall():
    fm = get_hub().fall_monitor
    if fm is None:
        return False
    return fm.inject_fall()


def measure_health():
    hm = get_hub().health_monitor
    if hm is None:
        return False
    return hm.request()


def _self_test():
    print("=" * 62)
    print("大白 3.3 硬件层自检")
    print("=" * 62)
    print("  模拟策略 BAYMAX_HW_SIM=%s（auto=没硬件就模拟）" % SIM_MODE)
    print("-" * 62)
    for k, v in probe().items():
        print("  %-10s %s" % (k, v))
    print("-" * 62)
    print("  跌倒状态机   %s" % _machine_self_test())
    print("  心率血氧算法 %s" % _ppg_self_test())
    print("-" * 62)
    print("说明：显示「缺失」的外设不影响语音链路，主程序照常运行。")
    return 0


def _machine_self_test():
    m = FallStateMachine(cooldown=0)
    seq = [(0.0, 1.0), (0.1, 0.3), (0.5, 3.2), (1.0, 1.0), (3.1, 1.0)]
    hits = [m.update(t, g) for t, g in seq]
    case1 = hits[-1] is True and sum(1 for h in hits if h) == 1

    m2 = FallStateMachine(cooldown=0)
    calm = [m2.update(i * 0.1, 1.0) for i in range(40)]
    case2 = not any(calm)

    m3 = FallStateMachine(cooldown=0)
    sit = [m3.update(0.0, 1.0), m3.update(0.2, 2.6), m3.update(0.5, 1.0),
           m3.update(3.0, 1.0)]
    case3 = not any(sit)

    ok = case1 and case2 and case3
    return "OK（跌倒检出 / 静止不误报 / 坐下不误报）" if ok else \
        "异常（检出=%s 静止=%s 坐下=%s）" % (case1, case2, case3)


def _ppg_self_test():
    if np is None:
        return "跳过（没有 numpy）"
    cases = [(60, 97), (72, 98), (88, 95), (110, 92)]
    lines, bad = [], 0
    for hr, spo2 in cases:
        red, ir = synth_ppg(hr=hr, spo2=spo2, seconds=8, seed=hr)
        res = analyze_ppg(red, ir)
        dh = abs(res["hr"] - hr) if res["hr"] else 999
        ds = abs(res["spo2"] - spo2) if res["spo2"] else 999
        good = dh <= 3 and ds <= 3
        bad += 0 if good else 1
        lines.append("%d/%d%%->%d/%d%%%s" % (hr, spo2, res["hr"], res["spo2"],
                                             "" if good else "✗"))
    red, ir = synth_ppg(hr=72, spo2=98, seconds=8, dc_ir=5000.0)
    no_finger = analyze_ppg(red, ir).get("finger") is False
    return ("OK（%s，无手指不误报）" if (not bad and no_finger) else "异常（%s，无手指=%s）") \
        % (" ".join(lines), no_finger)


def main(argv):
    if "--test" in argv:
        return _self_test()
    if "--ppg" in argv:
        print(_ppg_self_test())
        return 0
    if "--sim" in argv:
        os.environ["BAYMAX_HW_SIM"] = "1"
        print("强制模拟模式，跑一轮跌倒判定：")
        sim = SimMpu6050()
        m = FallStateMachine(cooldown=0)
        sim.inject_fall()
        t0 = time.monotonic()
        fired = False
        while time.monotonic() - t0 < 4.0:
            g = sim.read_magnitude()
            now = time.monotonic() - t0
            if m.update(now, g) and not fired:
                fired = True
                print("  %.2fs 判定跌倒，当时 g=%.2f" % (now, g))
            time.sleep(0.05)
        print("  结果：", "OK" if fired else "未触发（异常）")
        return 0 if fired else 1
    if "--sms" in argv:
        text = " ".join(a for a in argv if a != "--sms") or "大白测试短信"
        ok, msg = Ec20().send_sms(text)
        print("发送结果：", ok, msg)
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
