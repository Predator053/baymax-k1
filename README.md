# 大白 Baymax · 独居老人健康陪护与跌倒应急终端

面向 **K1（RISC-V）开发板** 的桌面式健康陪护终端：本地语音对话、健康知识问答、
身体数值记录、跌倒应急闭环、情绪关怀。全部推理在本地完成，断网可用。

比赛作品，包含语音主循环、外设驱动、闭环决策引擎、网页面板与实时终端。

## 核心特性

| 能力 | 说明 |
|---|---|
| 本地语音链路 | 离线唤醒（KWS）→ 离线识别（ASR）→ 声音克隆合成（ZipVoice） |
| 预存语音缓存 | 高频语句预生成为 WAV，按文本 SHA-256 精确命中，唤醒到出声约 **11 ms** |
| 三级意图 | 同一主题区分「是什么 / 为什么 / 怎么办」，答案互不相同 |
| 卫健委分级 | 血压按《中国高血压防治指南》分级并写入健康档案 |
| 跌倒闭环 | 先询问 → 可撤销 → 无人应答升级；摄像头抓拍 + 人脸计数二次确认 |
| 双通道告警 | 钉钉 markdown 卡片 + 4G 短信并行，含现场有人/无人、坐标、照片链接 |
| 自动拨号 | 20 秒 × 2 通，间隔 90 秒；两通未接再补发钉钉提醒 |
| 情绪关怀 | 静默 20 分钟或听到消极表达时主动关怀（语音 + 摄像头确认有人） |
| 网页面板 | KPI、时延分栏、硬件状态、现场画面、健康档案 |
| 实时终端 | 浏览器查看主程序终端输出（:8001），便于调试与演示 |

## 目录结构

```
src/
  baymax_3.3.py       语音主循环、知识检索、TTS 队列、跌倒与关怀闭环
  baymax_sensors.py   MPU6050 / EC20(4G) / CSI 摄像头 / GPS / 短信
  baymax_loop.py      风险评估与决策（L0–L3），可撤销告警
  baymax_health.py    口语数值解析（含中文数字）、卫健委分级、建档
  baymax_service.py   健康档案、用药提醒、面板指令通道
  baymax_panel.py     网页面板（KPI / 硬件 / 现场画面 / 健康档案）
  baymax_live.py      实时终端监视器（只读，浏览器查看终端输出）
  soft_i2c.py         软件 I2C（板载排针 I2C 未启用时使用）
docs/
  技术文档.pdf        架构、优化策略、部署说明、用户手册
  性能与测试报告.pdf   功能测试、时延、准确率、资源占用、异常处理
  演示脚本.md         现场演示流程与问答应答
scripts/
  自检.sh            演示前一键自检
```

## 硬件

| 部件 | 型号 | 接口 |
|---|---|---|
| 主控 | Spacemit K1-X MUSE Pi Pro（8 核，16 GB） | — |
| 六轴传感器 | MPU6050 | 软件 I2C（GPIO41/40，地址 0x68） |
| 4G 模块 | Quectel EC20 | USB / ModemManager（短信、定位、拨号） |
| 摄像头 | OV5647 | CSI3，1920×1080 |
| 音频输入 | USB 声卡（C-Media PCM2902） | USB，16 kHz 单声道 |
| 音频输出 | 蓝牙音响（A2DP，PipeWire） | 蓝牙 |

## 快速开始

```bash
# 1. 依赖
pip install numpy soundfile requests pyserial pillow

# 2. 语音模型
#    KWS：sherpa-onnx-kws-zipformer-wenetspeech-3.3M
#    TTS：ZipVoice（sherpa-onnx）+ 参考音频

# 3. 音频设备（输入输出可分开指定）
export BAYMAX_INPUT_DEVICE="plughw:0,0"      # USB 麦克风
export BAYMAX_OUTPUT_DEVICE="default"        # PipeWire → 蓝牙音响

# 4. 启动主程序（建议在 tmux 里跑，键盘快捷键才有效）
tmux new -s baymax
cd src && python3 -u baymax_3.3.py

# 5. 网页面板与实时终端
python3 baymax_panel.py 8000      # http://<板子IP>:8000
python3 baymax_live.py 8001       # http://<板子IP>:8001
```

## 配置

环境变量覆盖默认值：

| 变量 | 默认 | 说明 |
|---|---|---|
| `BAYMAX_INPUT_DEVICE` | 自动 | 录音设备，如 `plughw:0,0` |
| `BAYMAX_OUTPUT_DEVICE` | 自动 | 播放设备，如 `default` |
| `BAYMAX_KWS_THRESHOLD` | `0.10` | 唤醒灵敏度 |
| `BAYMAX_ESCALATE_SECONDS` | `45` | 告警升级等待 |
| `BAYMAX_CALL_NOFACE` | `1` | 是否自动拨号 |
| `BAYMAX_MOOD_IDLE_SECONDS` | `1200` | 静默多久触发情绪关怀 |
| `BAYMAX_MOOD_REQUIRE_FACE` | `1` | 关怀前是否要求摄像头看到人脸 |
| `BAYMAX_UI_META` | `1` | 是否显示时延明细 |
| `DINGTALK_WEBHOOK` | 无 | 钉钉机器人地址（需含关键词「紧急求助」） |
| `ALERT_PHONE` | 无 | 告警电话 | 
| `BAYMAX_CONFIG` | `config.json` | 位置与阈值配置 |

## 设计要点

**为什么先问再报？** 弯腰捡东西、坐下都可能产生类似跌落的加速度曲线。
直接报警会半夜惊动子女，老人几天后就把设备关了。先用一句话确认，
老人说「我没事」就本地终止，不惊动任何人。

**为什么摄像头只数人脸不识别表情？** 本项目使用 3 秒单帧、低光环境下
表情分类准确率不足以支撑告警决策，误判会直接损害可信度。摄像头只回答
「画面里有没有人」这个确定性问题，用于判断危险程度。

**为什么音频锁要区分输入输出？** 同一张声卡同时录放会冲突，需要互斥；
但输入（USB 麦克风）与输出（蓝牙音响）是两台设备时互不影响，
此时若仍抢占同一把锁，录音会把排队中的播报直接丢弃——表现为
「缓存命中了却没声音」。

## 已知限制

- 板载 ES8326 音频在现有固件下 I2S 时钟未使能，需外接 USB 麦克风与蓝牙音响
- 现场合成（缓存未命中）单句约 30 秒，高频语句需预生成缓存
- EC20 语音通道未接模拟音频，拨号仅作来电提醒，接通后无声
- 钉钉自定义机器人不支持直接发图，照片以链接形式提供
- GPS 室内无定位，坐标使用预设值

## 许可

MIT License，见 LICENSE。
