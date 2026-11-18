# Changelog

## 0.1.0 candidate

- Extract the 44-byte, 20-column sensor protocol, serial ownership and rolling ADC grid into a standalone package.
- Introduce sensor-scoped ROS 2 interfaces for raw rows, state, diagnostics and source metadata.
- Add a matrix viewer, neutral reference capture, a simulated sensor and sample-based replay.
- Add native RGB8 contact footprints, driver-owned unloaded references, recorded-reference replay and pixel-exact PNG export.
- Open the viewer on a soft-edged 300×300 contact estimate, with native grid inspection and per-mode PNG export.
