# Changelog

## 0.1.0 candidate

- Add a contact-print window for live and replay data with three source-timed exposures, manual capture, automatic release prints, and a labeled PNG export.
- Clear the live print at the first observed release and offer completed prints in a separate history view. Disable manual capture on an unloaded or indeterminate array.
- Render the latest contact print and its three source-timed exposures on the same acquisition-bound RGB response scale, with a labeled color legend and an unclipped peak readout.
- Show the contact footprint from the first supported observation. Trace each patch and its holes from noise-referenced native taxels, with three full response maps for comparison once the time history is available.
- Use a dark three-exposure imprint as the default contact print, with copper, cyan and ivory lines confined to the current measured footprint. Keep the calibrated color map available for viewing and export.

- Extract the 44-byte, 20-column sensor protocol, serial ownership and rolling ADC grid into a standalone package.
- Introduce sensor-scoped ROS 2 interfaces for raw rows, state, diagnostics and source metadata.
- Add a matrix viewer, neutral reference capture, a simulated sensor and sample-based replay.
- Add native RGB8 contact footprints, driver-owned unloaded references, recorded-reference replay and pixel-exact PNG export.
- Open on the 300×300 continuous contact view and support PNG export from the 20×20 scalar response.
- Use a shared blue-to-red intensity scale for the native ROS image, the continuous view and PNG exports. Show the scale in both contact views.
- Add a guided live known-load calibration, independent loading and unloading checks, release residuals, and same-load repeats with a local JSON report.
- Focus the live viewer on the contact surface and the load check on the loading/unloading comparison and release residual. Put diagnostics and exports in expandable details.
- Configure the live driver’s serial port and baud rate from the viewer, with passive discovery, connection status and explicit disconnect.
- Use a light blue-gray idle surface and a gradual low-response color ramp in the contact images, viewer and exports.
- Present the live contact surface in a square display with a shared intensity scale, measured peak readout and direct access to serial settings.
- Outline measured contact boundaries in the continuous viewer using the sensor’s per-cell unloaded thresholds, with a display toggle in Details.
- Add two-level full-face uniform-pressure calibration for individual taxel sensitivity, with independent validation, hardware and reference binding, and optional correction in both live contact views.
