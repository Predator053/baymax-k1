#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time

import gpiod
from gpiod.line import Bias, Direction, Drive, Value

CHIP = "/dev/gpiochip0"
SDA = 41
SCL = 40
HALF = 6e-6


class SoftI2C:
    def __init__(self, sda=SDA, scl=SCL, chip=CHIP, half=HALF):
        self.sda, self.scl, self.half = sda, scl, half
        self._out = {
            sda: gpiod.LineSettings(direction=Direction.OUTPUT,
                                    output_value=Value.ACTIVE,
                                    drive=Drive.OPEN_DRAIN,
                                    bias=Bias.PULL_UP),
            scl: gpiod.LineSettings(direction=Direction.OUTPUT,
                                    output_value=Value.ACTIVE,
                                    drive=Drive.OPEN_DRAIN,
                                    bias=Bias.PULL_UP),
        }
        self._in_sda = gpiod.LineSettings(direction=Direction.INPUT,
                                          bias=Bias.PULL_UP)
        self.req = gpiod.request_lines(chip, consumer="soft-i2c", config=self._out)
        self.set_sda(1)
        self.set_scl(1)

    def set_sda(self, v):
        self.req.set_value(self.sda, Value.ACTIVE if v else Value.INACTIVE)

    def set_scl(self, v):
        self.req.set_value(self.scl, Value.ACTIVE if v else Value.INACTIVE)

    def get_sda(self):
        return 1 if self.req.get_value(self.sda) == Value.ACTIVE else 0

    def _d(self):
        time.sleep(self.half)

    def start(self):
        self.set_sda(1); self.set_scl(1); self._d()
        self.set_sda(0); self._d()
        self.set_scl(0); self._d()

    def stop(self):
        self.set_sda(0); self._d()
        self.set_scl(1); self._d()
        self.set_sda(1); self._d()

    def _write_bit(self, b):
        self.set_sda(1 if b else 0); self._d()
        self.set_scl(1); self._d()
        self.set_scl(0); self._d()

    def _read_bit(self):
        self.req.reconfigure_lines({self.sda: self._in_sda})
        self._d()
        self.set_scl(1); self._d()
        v = self.get_sda()
        self.set_scl(0); self._d()
        self.req.reconfigure_lines({self.sda: self._out[self.sda]})
        self.set_sda(1)
        self._d()
        return v

    def _write_byte(self, byte):
        for i in range(7, -1, -1):
            self._write_bit((byte >> i) & 1)
        return self._read_bit()

    def _read_byte(self, ack=True):
        v = 0
        for _ in range(8):
            v = (v << 1) | self._read_bit()
        self._write_bit(0 if ack else 1)
        return v

    def write(self, addr, reg, data):
        self.start()
        if self._write_byte(addr << 1):
            self.stop(); raise IOError("地址 0x%02X 无应答" % addr)
        self._write_byte(reg)
        for b in data:
            self._write_byte(b)
        self.stop()

    def read(self, addr, reg, n):
        self.start()
        if self._write_byte(addr << 1):
            self.stop(); raise IOError("地址 0x%02X 无应答" % addr)
        self._write_byte(reg)
        self.start()
        if self._write_byte((addr << 1) | 1):
            self.stop(); raise IOError("地址 0x%02X 读无应答" % addr)
        out = [self._read_byte(ack=(i < n - 1)) for i in range(n)]
        self.stop()
        return out

    def probe(self, addr):
        self.start()
        nak = self._write_byte(addr << 1)
        self.stop()
        return nak == 0

    def read_byte_data(self, addr, reg):
        return self.read(addr, reg, 1)[0]

    def write_byte_data(self, addr, reg, val):
        self.write(addr, reg, [int(val) & 0xFF])

    def read_i2c_block_data(self, addr, reg, n):
        return self.read(addr, reg, n)

    def close(self):
        try:
            self.req.release()
        except Exception:
            pass


def main():
    bus = SoftI2C()
    try:
        if len(sys.argv) > 1:
            addr = int(sys.argv[1], 0)
            print("探测 0x%02X …" % addr)
            if bus.probe(addr):
                print("  ✅ 有应答")
                who = bus.read(addr, 0x75, 1)[0]
                print("  WHO_AM_I = 0x%02X（MPU6050 正常是 0x68）" % who)
                ax = bus.read(addr, 0x3B, 6)
                print("  加速度原始值:", ax)
            else:
                print("  ❌ 无应答")
            return
        print("扫描 0x08 ~ 0x77 …")
        found = []
        for a in range(0x08, 0x78):
            try:
                if bus.probe(a):
                    found.append(a)
                    print("  ✅ 0x%02X" % a, flush=True)
            except Exception as e:
                print("  0x%02X 异常 %s" % (a, e))
        print("结果:", ["0x%02X" % x for x in found] or "没有设备")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
