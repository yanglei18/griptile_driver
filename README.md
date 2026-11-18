# griptile_driver

适用于单块 USB 串口夹爪触觉阵列的 ROS 2 驱动和二维查看器。默认窗口把 20×20 个触点显示成一张柔和的接触印迹图，可观察夹爪与物体的接触位置、范围和相对强弱。点击单元可查看 ADC 曲线。查看器、录制器与其他 ROS 节点可以同时订阅同一驱动。[English guide](docs/README.en.md)

![同一组模拟触点的原生颜色与柔和接触印迹](docs/contact-surface-comparison.png)

[实际窗口截图](docs/viewer-demo.png) · [柔化前后对照](docs/contact-surface-softening.png) · [原生 20×20 PNG 示例](docs/contact-ring.png)

## 构建

环境为 Ubuntu 24.04、ROS 2 Jazzy、colcon 和已初始化的 rosdep。在仓库根目录运行以下命令。

```bash
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src --rosdistro jazzy -y
colcon build --base-paths src
source install/setup.bash
```

每个新终端先加载上述两个 setup.bash。只部署采集节点的主机可以将 rosdep 的包目录设为 `--from-paths src/griptile_{core,msgs,driver}`，并给 colcon 增加 `--packages-up-to griptile_driver`，启动参数 `gui` 默认关闭。

## 打开二维查看器

没有硬件时，从仓库根目录启动模拟器。演示先在专用虚拟串口上提供无载数据，收到驱动的参考就绪消息后循环展示环形接触、松开和双点接触。话题位于 `/tactile/demo`。按 Ctrl+C 退出。

```bash
python3 examples/demo.py
```

使用夹爪时，把下面的端口换成操作系统中这块阵列对应的路径。建议选用 `/dev/serial/by-id/` 中稳定的设备链接。默认波特率为 115200。同一块板由一个驱动进程打开，多个应用可订阅它的话题。

```bash
ros2 launch griptile_driver sensor.launch.py port:=/dev/ttyUSB0 gui:=true
```

窗口默认选择「柔和接触图（估算）」。让夹爪张开、感应面保持无载，点击一次「采集无载参考（3 秒）」，等有效单元变白后即可接触物体。每次建立参考的是驱动，同一传感器的所有窗口和 ROS 图像订阅者共享该结果。

| 像素颜色 | 含义 |
| --- | --- |
| 白色 | 该点有有效测量，变化处于无载噪声范围 |
| 暗红褐色至鲜红 | 接触响应由弱到强，超过所选量程后保持纯红 |
| 灰色 | 该点缺少可用参考或新鲜读数 |

默认视图将相邻触点的响应画成 300×300 的连续色面，接触边缘逐渐淡入白底，边缘颜色会随覆盖程度变浅。点击「导出估算图」可保存这张 PNG。左上角的「接触图 RGB」显示一触点一像素的原生图，使用最近邻放大并导出 20×20 PNG。两种视图共享无载参考，点击读数、曲线与接触点数均取自 400 个实测触点。第 0 行在上，第 0 列在左。

颜色表示相对无载的 ADC 变化。换算牛顿、帕斯卡或平方毫米需要对应的力学标定或几何尺寸。

显示使用固定色域，默认去噪后 3000 ADC 对应满红。启动时传入 `contact_span_adc:=6000.0` 可增大显示量程。默认 `contact_direction:=both` 同时接受 ADC 向上或向下的变化，已知夹爪受压极性时可选 `increase` 或 `decrease`。原始 ADC、有符号 ΔADC 和快照 CSV 均可用于诊断。已核验 USB 身份的设备支持保存和复用无载参考。

无界面运行时也可以在夹爪张开的状态下采集参考。

```bash
ros2 service call /tactile/sensor_0/set_contact_reference griptile_msgs/srv/SetContactReference '{}'
```

## ROS 接口与时间

默认命名空间为 `/tactile/sensor_0`。启动时可使用 `sensor_id:=jaw_a` 指定传感器逻辑名称，对应默认命名空间 `/tactile/jaw_a`。

| 相对话题 | 消息 | 用途 |
| --- | --- | --- |
| `samples` | `griptile_msgs/msg/TactileRow44Array` | 按串口顺序保存的 ADC 行、主机接收时刻、序号和丢批计数。推荐作为录制输入 |
| `info` | `griptile_msgs/msg/SensorInfo` | 协议、布局、单位、当前连接及可用的设备标识。支持晚加入的订阅者 |
| `state` | `griptile_msgs/msg/TactileGridState` | 供显示的滚动阵列，含逐行时间、有效掩码和更新掩码。默认 30 Hz |
| `diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | 观察连接、数据龄期、丢批和结构错误 |
| `contact_image` | `sensor_msgs/msg/Image` | 默认发布的原生接触图。20×20、`rgb8`、每行 60 字节，header 沿用源阵列时间 |
| `contact_response` | `sensor_msgs/msg/Image` | 与原生接触图共用源时间，20×20、`32FC1`。值为去噪后的 ADC 变化幅值，0 为未触发，NaN 为缺测或无参考，保留超过显示量程的幅值 |
| `contact_reference` | `griptile_msgs/msg/ContactReference` | 参考采集进度、设备和会话、就绪后的逐点基线及阈值、色域和极性。支持晚加入的订阅者 |
| `heatmap` | `sensor_msgs/msg/Image` | 可选原始 ADC 彩色预览，通过 `publish_heatmap:=true` 启用 |

每行 44 字节，由帧头 `A5 5A`、一个大端 16 位行号和 20 个大端 16 位 ADC 值组成。协议无已知校验和、硬件时间戳或硬件序号。源时间来自串口读取完成的主机时钟。矩阵由先后收到的行组成，默认超过 100 ms 的行会标记过期。处理算法应同时读取 `adc` 和 `valid_mask`，零本身是合法读数。

## 录制与回放

在默认传感器运行时，从另一终端录制采样、设备信息及接触参考。待录制器显示已订阅后开始采集，按 Ctrl+C 完成保存。自定义命名空间时，同时修改话题名称和 QoS 文件中的键。

```bash
qos="$(ros2 pkg prefix --share griptile_driver)/config/record_qos.yaml"
ros2 bag record -s mcap -o grip_run --qos-profile-overrides-path "$qos" \
  --topics /tactile/sensor_0/samples /tactile/sensor_0/info /tactile/sensor_0/contact_reference
```

停止该采集驱动，在一个终端启动回放窗口，再在另一个终端播放刚录好的包。重建的阵列位于 `/tactile/replay/state`。播放结束后，只要回放节点仍在运行，新打开的查看器也能通过 `/tactile/replay/info` 获取原传感器信息。录制时若修改过 `max_cell_age_ms`，回放时传入相同的值即可复用相同的过期判断。

```bash
ros2 launch griptile_driver replay.launch.py gui:=true
```

```bash
ros2 bag play grip_run --clock 100 --delay 1
```

新录包中的参考可用于恢复会话的接触图，查看 `/tactile/replay/contact_image` 即可。旧包缺少参考时仍能查看原始 ADC；若有明确的空载保持段，可以从该段采集参考。需要逐帧保存当时发布的精确像素时，录制命令中再加入 `/tactile/sensor_0/contact_image`，播放后可直接订阅该原始图像话题。

若录制的传感器名为 `jaw_a`，回放启动命令也要增加 `sensor_id:=jaw_a`。若录制时还单独指定了命名空间，例如 `/robot/jaw`，同时增加 `source_namespace:=/robot/jaw`。

## 验证与维护

```bash
QT_QPA_PLATFORM=offscreen python3 -m pytest -q --import-mode=importlib src/*/test tests
python3 scripts/compare_wire.py
```

测试会通过虚拟串口向正式节点供数，录制 MCAP，再从行采样重建阵列。默认输入是确定性生成的样例。设置 `GRIPTILE_WIRE` 可用自己的完整、行对齐的 44 字节协议记录运行 `tests/test_sample_roundtrip.py`。离线对照工具接受 `--wire` 和可选的 `--baseline-source`，用于对照含指定版本的旧工程。

已使用一段 41,932 字节的历史记录对照提取前后的实现。953 行文件解析相同；通过虚拟串口重录的 19,060 个 ADC 值逐项吻合，回放重建的 400 个单元、有效位和行时间与采集端一致。实时解析器使用下一行帧头确认上一行，文件自然结束时最后 44 字节保持待确认。本次串口验收额外送入 4 字节下一帧头，让最后一行参与比较。历史文件保存了字节顺序，原始 USB 到达间隔未被记录。这项验证覆盖数据链路，实体夹爪的安装朝向、受力响应和持续运行需另行实测。公开仓库提供合成样例和验证摘要。[新旧对照结果](tests/evidence/wire-parity.json) · [ROS 录制回放结果](tests/evidence/wire-ros-roundtrip.json)

| 包 | 职责 |
| --- | --- |
| `griptile_core` | 串口访问、行协议解析、逐行时间和有效性、无载参考 |
| `griptile_msgs` | 采样、阵列状态与设备信息的 ROS 消息定义 |
| `griptile_driver` | 采集、诊断、录制配置与回放节点 |
| `griptile_viewer` | 订阅 ROS 消息，显示与导出二维阵列 |

采集内核源自 VisionTactile-Dataset 的 `vt_tactile_capture`，提取基线为 `74e861c23aa54d104503c6e521d739ec25163c2b`。本项目延用 Apache-2.0 许可证，可在独立工作区构建与安装。[许可证](LICENSE)
