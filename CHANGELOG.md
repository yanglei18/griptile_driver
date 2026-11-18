# Changelog

## 0.1.0 candidate

- Extract the 44-byte, 20-column sensor protocol, serial ownership and rolling ADC grid into a standalone package.
- Introduce sensor-scoped ROS 2 interfaces for raw rows, state, diagnostics and source metadata.
- Add a matrix viewer, neutral reference capture, a simulated sensor and sample-based replay.
- Add native RGB8 contact footprints, driver-owned unloaded references, recorded-reference replay and pixel-exact PNG export.
- Open on the 300×300 continuous contact view and support PNG export from the 20×20 scalar response.
- Use a shared blue-to-red intensity scale for the native ROS image, the continuous view and PNG exports. Show the scale in both contact views.
- Add a guided live known-load calibration, independent loading and unloading checks, release residuals, and same-load repeats with a local JSON report.
- Focus the live viewer on the contact surface and the load check on the loading/unloading comparison and release residual. Put diagnostics and exports in expandable details.
- Configure the live driver’s serial port and baud rate from the viewer, with passive discovery, connection status and explicit disconnect.
