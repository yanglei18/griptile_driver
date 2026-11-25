# Changelog

## 0.1.0 candidate

- Add a contact-print page for live and replay data with three source-timed exposures, manual capture, automatic release prints, and a labeled PNG export.
- Clear the live print at the first observed release and offer completed prints in a separate history view. Disable manual capture on an unloaded or indeterminate array.
- In the optional response style, render the contact print and its three source-timed exposures on the same acquisition-bound RGB scale, with a labeled color legend and an unclipped peak readout.
- Show the contact footprint from the first supported observation. Trace each patch and its holes from noise-referenced native taxels. Live and saved prints open on the main image. Select **三帧对照** to expand the three response maps on the same page. Automatic capture respects this choice, and PNG exports always include the three maps.
- Use a dark three-exposure imprint as the default contact print, with copper, cyan and ivory lines confined to the current measured footprint. Keep the calibrated color map available for viewing and export.
- Switch between the contact print and the array controls in the same main window. Use `imprint:=true` to start on the print page during acquisition or replay. Automatic print capture continues on the array page.
- Keep the contact print at a fixed full-array scale as the footprint changes and the three exposures arrive. Manual zoom locks the selected region for live viewing, history and PNG export.

- Extract the 44-byte, 20-column sensor protocol, serial ownership and rolling ADC grid into a standalone package.
- Introduce sensor-scoped ROS 2 interfaces for raw rows, state, diagnostics and source metadata.
- Add a matrix viewer, neutral reference capture, a simulated sensor and sample-based replay.
- Add native RGB8 contact footprints, driver-owned unloaded references, recorded-reference replay and pixel-exact PNG export.
- Open on a fixed 300×300 contact map. Measured patches progress from blue through purple to red on a dark blue-black unloaded surface. A patch must meet its taxels’ noise-based detection criteria. The optional outline follows the same patch and preserves measured holes and gaps. The PNG exports the colored surface.
- Keep the native 20×20 RGB view, ROS image and native PNG on the light blue-gray to red scale. Weaker individual responses can be inspected in this view.
- Place unloaded zeroing (**空载归零**) and contact print (**触印**) beside the main map. Details now groups serial settings, taxel calibration, known-load checks, display modes and exports. The main map guides connection and zeroing and reports progress and failures.
- Save manually acquired or imported unloaded references on the live acquisition host for verified devices. Reuse a reference on reconnect when the sensor ID, hardware identity, protocol, layout and response direction match. Replay uses its recorded reference.
- Add a guided live known-load calibration, independent loading and unloading checks, release residuals, and same-load repeats with a local JSON report.
- Focus the live viewer on the contact surface and the load check on the loading/unloading comparison and release residual. Put diagnostics and exports in expandable details.
- Configure the live driver’s serial port and baud rate from the viewer, with passive discovery, connection status and explicit disconnect.
- Add two-level full-face uniform-pressure calibration for individual taxel sensitivity, with independent validation, hardware and reference binding, and optional correction in both live contact views.
