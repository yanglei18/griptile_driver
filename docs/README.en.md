# griptile_driver

A ROS 2 driver and a 2D viewer for one USB serial gripper tactile array. Each taxel maps to one RGB pixel. See where the gripper touches an object, how many cells respond and how response varies across its contact footprint. The driver owns the serial device and ROS subscribers share its output. [中文说明](../README.md)

![Simulated release, light and firm contact on the same ring, and two separate pads on the continuous display](contact-footprints.png)

[Native 20×20 RGB example](contact-ring.png) · [Qt window](viewer-demo.png)

## Build and run

Use Ubuntu 24.04 with ROS 2 Jazzy, colcon and initialized rosdep. Run these commands from the repository root.

```bash
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src --rosdistro jazzy -y
colcon build --base-paths src
source install/setup.bash
```

Source both setup files in each new terminal. Run the full driver and viewer on a simulated sensor without hardware. Press Ctrl+C to stop.

```bash
python3 examples/demo.py
```

To use a physical array, launch the driver and viewer, open **串口设置** (Serial settings), choose a device on the acquisition host or type its path, and click **连接** (Connect). The default baud rate is 115200. The dialog lists Linux by-id aliases when available and distinguishes an open port from a sensor streaming valid data. [Serial settings window](serial-settings.png)

```bash
ros2 launch griptile_driver sensor.launch.py gui:=true
```

The dialog can refresh the list, switch ports and disconnect. Changing connections starts a new acquisition session. You can also select a port at startup with `port:=/dev/ttyUSB0 baudrate:=115200`. Omit `gui:=true` for headless acquisition. On a headless installation use `--from-paths src/griptile_{core,msgs,driver}` with rosdep and build with `colcon build --base-paths src --packages-up-to griptile_driver`.

Expand **详情** for view selection, cell traces and exports. The array uses row and column addresses 0–19, with row zero at the top and column zero at the left. Open the gripper, unload the surface and click the three-second reference button. Once the reference is ready, light blue-gray pixels indicate measured cells with no detected contact. The first positive color step is one RGB code from the background. Increasing response moves through pale blue to cyan at 20% of the display span, then green, yellow, orange and red. Gray pixels indicate missing reference or stale data. The demo supplies unloaded samples until the driver confirms the reference, then cycles through a ring, release and two separate contact patches.

The live viewer uses source-matched scalar responses to color both contact views. Archived RGB images retain their original colors. [Compare the same inputs from zero to light contact](contact-onset.png).

The native **接触图 RGB** view magnifies the pixels with nearest-neighbor sampling. Contact-mode PNG export writes the original 20×20 RGB image. The contact footprint counts responding cells; conversion to physical area requires sensor dimensions. Intensity is a noise-gated ADC change; force or pressure units require physical calibration. Raw ADC and signed delta views, per-cell traces and CSV export remain available. Saved references can be reused with a verified matching device and layout.

The default **连续接触面（估算）** view displays the same 400 taxels as a 300×300 RGB surface. The viewer interpolates noise-gated ADC amplitudes between neighboring taxel centers and estimates the contact outline from their active flags. Unknown regions stay gray. Colors at all 400 measured centers match the native grid. Picking, time series and contact counts still use the original addresses. The surface view exports a 300×300 PNG. Both views share the same unloaded reference.

![Native and interpolated views of the same simulated ring and three broad ridges](contact-surface-comparison.png)

The three RGB channels encode one tactile response per taxel. The same response always receives the same color across frames. The red endpoint defaults to 3000 noise-gated ADC counts. Set `contact_span_adc:=6000.0` to extend the range. The default `contact_direction:=both` accepts either sign of ADC change. Select `increase` or `decrease` when the pressure polarity is known. All subscribers share the driver's calibration. From a headless terminal with the gripper unloaded, request it with the command below.

```bash
ros2 service call /tactile/sensor_0/set_contact_reference griptile_msgs/srv/SetContactReference '{}'
```

### Check the display against a known load

Align a force gauge with the normal to the sensing face and keep the indenter and its position fixed. Open **已知载荷验证** in the live viewer. Reuse the current unloaded reference, or acquire one with the jaw open. Enter two gauge readings in newtons, a lower independent check load and a higher calibration load.

| Hold | Gauge reading | Use |
| --- | --- | --- |
| Load up | Lower check load | Independent reading during clamping |
| Load further | Higher calibration load | Fit a single gain using this point and the existing zero |
| Unload partway | Same lower check load | Check the estimate on unloading against the gauge and the loading reading |
| Release | Zero | Measure the residual after release |

Follow the single action shown for each step and capture when the gauge reaches the target. The high calibration load requires a stable hold. Independent checks retain the first complete measurement window and its variation, including fluctuations on release. A gap in acquisition restarts the current sampling window. The window estimates total normal force on this sensor face from the sum of the 400 native noise-gated responses. The RGB image keeps its ADC scale. A cell above the red endpoint still contributes its full response to the force estimate. The gain is local to this setup. Accuracy between the two fit endpoints must be tested independently. A per-cell pressure scale in kPa requires additional calibration.

The result compares the two peak colors and force estimates at the same gauge reading and shows the residual after release. Open the details to inspect individual signed errors and export the JSON report. To compare locations or indenters, apply the same check load and use **同载荷复测**. Every repeat uses the original gain. With the default bidirectional contact detector, reverse electrical excursions can also show up as contact on release. The signed response in the report helps identify that behavior. [Example verification window with simulated loads](known-load-check.png)

## ROS interfaces

The default namespace is `/tactile/sensor_0`. Set a different logical ID with `sensor_id:=jaw_a` to use `/tactile/jaw_a`. The demo runs in `/tactile/demo`.

| Topic | Type | Data |
| --- | --- | --- |
| `samples` | `griptile_msgs/msg/TactileRow44Array` | Decoded ADC rows in wire order, host read times, software sequence numbers and loss counters. Record this stream |
| `info` | `griptile_msgs/msg/SensorInfo` | Protocol, layout, units and available device identity. Uses reliable transient-local QoS |
| `state` | `griptile_msgs/msg/TactileGridState` | Rolling array, row times, valid and updated masks. Defaults to 30 Hz |
| `diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | Connection, freshness, dropped batches and framing errors |
| `contact_image` | `sensor_msgs/msg/Image` | Native 20×20 RGB8 contact image, 60 bytes per row, source header preserved |
| `contact_response` | `sensor_msgs/msg/Image` | Native 20×20 `32FC1` noise-gated ADC amplitudes, paired to the RGB image by its acquisition header. Zero indicates inactive, NaN indicates unknown. Values above the display range are retained |
| `contact_reference` | `griptile_msgs/msg/ContactReference` | Reference progress, session and device binding, ready coefficients, direction and display range. Transient-local |
| `heatmap` | `sensor_msgs/msg/Image` | Optional raw ADC color preview. Enable with `publish_heatmap:=true` |

The live driver exposes the `manage_serial` service (`griptile_msgs/srv/ManageSerial`). Use `refresh` to read the acquisition host’s port inventory, current configuration and revision. Send the observed `expected_revision` with a subsequent `connect` or `disconnect` request.

A 44-byte packet has a two-byte `A5 5A` header, a big-endian uint16 row index and twenty big-endian uint16 ADC values. The protocol has no documented checksum, device timestamp or hardware sequence counter. Samples use host read-completion times. A grid combines sequential row updates and uses a 100 ms default freshness limit. Read the validity mask alongside ADC values. Zero can be a valid measurement.

## Record and replay

With the default driver running, start the recorder in another terminal, wait for the subscriptions and press Ctrl+C to finish recording. For a custom namespace, update both topic names and the corresponding YAML keys.

```bash
qos="$(ros2 pkg prefix --share griptile_driver)/config/record_qos.yaml"
ros2 bag record -s mcap -o grip_run --qos-profile-overrides-path "$qos" \
  --topics /tactile/sensor_0/samples /tactile/sensor_0/info /tactile/sensor_0/contact_reference
```

Stop the live driver, launch the replay node and viewer in one terminal, then play the bag from another. The reconstructed array is published on `/tactile/replay/state`. While the replay node remains running, late subscribers can read the original sensor metadata on `/tactile/replay/info` after the player exits. If live acquisition used a custom `max_cell_age_ms`, set the same value on replay to apply the same freshness policy.

```bash
ros2 launch griptile_driver replay.launch.py gui:=true
```

```bash
ros2 bag play grip_run --clock 100 --delay 1
```

The replay node uses recorded references on the matching acquisition session to reconstruct `/tactile/replay/contact_image`. Older bags without a reference can display raw ADC. To create a reference during replay, use a known unloaded interval. To preserve the exact RGB frames as originally published, also record `/tactile/sensor_0/contact_image` and view that original topic during playback.

To replay a recording made with `sensor_id:=jaw_a`, pass that same argument to the replay launch. If you also used a custom namespace such as `/robot/jaw`, add `source_namespace:=/robot/jaw`.

## Verification

```bash
QT_QPA_PLATFORM=offscreen python3 -m pytest -q --import-mode=importlib src/*/test tests
python3 scripts/compare_wire.py
```

Tests cover decoding, fragmented input, rolling state, serial connection boundaries, ROS messages, Qt and an end-to-end MCAP round trip through a virtual serial port. CI uses generated data. Set `GRIPTILE_WIRE` to a complete, row-aligned row44 capture to exercise `tests/test_sample_roundtrip.py` with your own bytes. The offline comparison script accepts `--wire` and optionally `--baseline-source` pointing to a trusted checkout containing the pinned upstream commit.

A 41,932-byte archived sensor trace contained 953 rows. File decoding matched the previous implementation. All 19,060 ADC values survived recording through the installed driver. The replayed 400-cell grid, valid mask and row timestamps matched acquisition. The live parser confirms a row on receiving the next header, so an unextended file leaves the final 44 bytes pending. The round-trip test supplies four extra header bytes to confirm that row. The archived input preserves byte order but lacks the original USB arrival times. The checked-in reports cover the software path. Gripper mounting orientation, mechanical response and endurance require physical testing. [Extraction comparison](../tests/evidence/wire-parity.json) · [ROS round trip](../tests/evidence/wire-ros-roundtrip.json)

## Package layout

| Package | Responsibility |
| --- | --- |
| `griptile_core` | Serial access, wire decoding, row timing and freshness, unloaded reference |
| `griptile_msgs` | ROS 2 sample, state and metadata message definitions |
| `griptile_driver` | Live acquisition, diagnostics, recording configuration and replay |
| `griptile_viewer` | Standalone Qt subscriber for matrix inspection and export |

The acquisition core derives from VisionTactile-Dataset's `vt_tactile_capture` at commit `74e861c23aa54d104503c6e521d739ec25163c2b`. This workspace builds and runs independently. [Apache-2.0 license](../LICENSE)
