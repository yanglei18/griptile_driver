"""One reference and one native contact image for all consumers of a sensor."""

import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image

from griptile_core.contact_image import DEFAULT_CONTACT_SPAN, contact_rgb
from griptile_core.display_response import ReferenceProfile, ResponseSettings, TactileResponseProcessor
from griptile_core.grid import COLS, LAYOUT_ID, ROWS
from griptile_core.protocol import PROTOCOL_ID
from griptile_msgs.msg import ContactReference
from griptile_msgs.srv import SetContactReference


class ContactImaging:
    """Own contact detection, source-bound calibration and presentation together."""

    def __init__(self, node, sensor_id, *, image_events=None):
        self.sensor_id = sensor_id
        startup = ParameterDescriptor(read_only=True)
        span = node.declare_parameter("contact_span_adc", DEFAULT_CONTACT_SPAN, startup).value
        direction = node.declare_parameter("contact_direction", "both", startup).value
        self.auto_capture = node.declare_parameter("contact_auto_reference", False, startup).value
        self.settings = ResponseSettings(span, direction)
        self.processor = TactileResponseProcessor(self.settings)
        self.recorded_phase = None
        self.recorded_hardware_id = ""
        self.state = None
        self.info = None
        self.started_auto = False
        self._published_reference = None
        self.image_publisher = node.create_publisher(
            Image, "contact_image", 1, event_callbacks=image_events)
        self.reference_publisher = node.create_publisher(
            ContactReference, "contact_reference",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.service = node.create_service(
            SetContactReference, "set_contact_reference", self.set_reference)
        self._report()

    @property
    def identity(self):
        state = self.state
        return (state.session_id, state.stream_epoch) if state and state.session_id else None

    @property
    def hardware_id(self):
        info = self.info
        if (info is not None and info.connected and info.sensor_id == self.sensor_id
                and self.identity == (info.session_id, info.stream_epoch)
                and info.protocol_id == PROTOCOL_ID and info.layout_id == LAYOUT_ID
                and info.rows == ROWS and info.cols == COLS and info.unit == "adc_counts"):
            return info.hardware_id
        return ""

    def update_info(self, info):
        previous = self.hardware_id
        self.info = info
        if ((previous and previous != self.hardware_id)
                or (self.recorded_hardware_id and self.hardware_id
                    and self.recorded_hardware_id != self.hardware_id)):
            self._clear_reference("hardware_changed")
            if self.state is not None:
                self._publish_pixels(contact_rgb({"reference_ready": False}))
        self._report()

    def _clear_reference(self, reason):
        self.processor.reset(reason)
        self.processor.configure(display_range_adc=self.settings.display_range_adc, direction=self.settings.direction)
        self.recorded_phase = None
        self.recorded_hardware_id = ""

    def reset_timeline(self, reason):
        """Invalidate the latched reference even if a seek never produces another row."""
        self._clear_reference(reason)
        if self.state is not None:
            self._publish_pixels(contact_rgb({"reference_ready": False}))
        self.state = None
        self.info = None
        self.started_auto = False
        self._report()

    def _restore_recorded(self, reference):
        # The replay queue matches acquisition session, epoch and timestamp.
        # This recovers observations made in *that capture*, including PTYs;
        # cross-session file imports must have a verified hardware ID.
        if self.hardware_id and reference.hardware_id and self.hardware_id != reference.hardware_id:
            raise ValueError("Recorded contact reference hardware mismatch")
        settings = ResponseSettings(reference.span_adc, reference.direction)
        if reference.phase == "ready":
            profile = ReferenceProfile(tuple(reference.baseline_adc),
                                       tuple(reference.activation_threshold_adc), tuple(reference.valid_mask))
            self.processor.load_reference(profile, self.identity, self.state.snapshot_monotonic_ns)
            self.recorded_phase = None
        else:
            if (reference.phase not in ("unreferenced", "collecting", "failed")
                    or not 0 <= reference.progress <= 1
                    or len(reference.baseline_adc) or len(reference.activation_threshold_adc) or len(reference.valid_mask)):
                raise ValueError("Invalid recorded reference status")
            self.processor.reset(reference.reason)
            self.recorded_phase = reference.phase, reference.progress, reference.reason
        self.processor.configure(display_range_adc=settings.display_range_adc, direction=settings.direction)
        self.started_auto = True
        self.recorded_hardware_id = reference.hardware_id
        self._published_reference = None

    def update(self, state, recorded_reference=None):
        previous_identity = self.identity
        self.state = state
        if self.identity != previous_identity:
            self._clear_reference("stream_changed")
            self.started_auto = False
        if recorded_reference is not None and state.connected and self.identity:
            try:
                self._restore_recorded(recorded_reference)
            except ValueError as exc:
                self._clear_reference(str(exc))
        if not self.identity or not state.connected:
            self._clear_reference("disconnected")
        elif not state.time_valid or not any(state.valid_mask):
            self.processor.clear_activity()
            self.processor.cancel_reference(state.reason or "no_fresh_samples")
        else:
            now = state.snapshot_monotonic_ns
            if self.auto_capture and not self.started_auto:
                self.processor.begin_reference(self.identity, now)
                self.started_auto = True
            if self.processor.status.phase == "collecting":
                stamps = np.repeat(state.row_monotonic_ns, COLS)
                self.processor.observe(state.adc, state.valid_mask, stamps, self.identity, now)
        self._present()

    def set_reference(self, request, result):
        try:
            if not self.identity or not self.state.connected or not self.state.time_valid or not any(self.state.valid_mask):
                raise ValueError("No fresh sensor rows available")
            target = (request.session_id, request.stream_epoch)
            if (request.sensor_id or request.session_id or request.stream_epoch or request.hardware_id
                    or request.load_profile):
                if request.sensor_id != self.sensor_id or target != self.identity:
                    raise ValueError("Sensor session or epoch changed")
                if request.hardware_id and request.hardware_id != self.hardware_id:
                    raise ValueError("Hardware identity mismatch")
            now = self.state.snapshot_monotonic_ns
            if request.load_profile:
                if not request.hardware_id or request.hardware_id != self.hardware_id:
                    raise ValueError("Loading a reference requires a verified matching device")
                profile = ReferenceProfile(tuple(request.baseline_adc),
                                           tuple(request.activation_threshold_adc),
                                           tuple(request.valid_mask))
                self.processor.load_reference(profile, self.identity, now)
            else:
                if len(request.baseline_adc) or len(request.activation_threshold_adc) or len(request.valid_mask):
                    raise ValueError("Profile data requires load_profile=true")
                if self.processor.status.phase == "collecting":
                    raise ValueError("Reference collection already in progress")
                self.processor.begin_reference(self.identity, now)
            self.recorded_hardware_id = ""
            self.started_auto = True
            self.recorded_phase = None
            self._published_reference = None  # A ready profile can directly replace another.
            self._present()
        except ValueError as exc:
            result.success, result.message = False, str(exc)
        else:
            result.success, result.message = True, self.processor.status.phase
        return result

    def _present(self):
        state = self.state
        if self.identity and state.connected and state.time_valid:
            response = self.processor.transform(
                state.adc, state.valid_mask, self.identity, state.snapshot_monotonic_ns)
        else:
            response = {"reference_ready": False}
        self._report()
        self._publish_pixels(contact_rgb(response))

    def _publish_pixels(self, pixels):
        self.image_publisher.publish(Image(
            header=self.state.header, height=ROWS, width=COLS, encoding="rgb8",
            step=COLS * 3, data=pixels.tobytes()))

    def _report(self):
        status = self.processor.status
        phase, progress, reason = self.recorded_phase or (status.phase, status.progress, status.reason)
        key = (self.identity, self.hardware_id, phase, progress, reason)
        if key == self._published_reference:
            return
        msg = ContactReference(
            sensor_id=self.sensor_id, hardware_id=self.hardware_id, phase=phase,
            progress=progress, reason=reason, span_adc=self.processor.settings.display_range_adc,
            direction=self.processor.settings.direction)
        if self.state is not None:
            msg.header = self.state.header
            msg.session_id = self.state.session_id
            msg.stream_epoch = self.state.stream_epoch
        if status.phase == "ready":
            profile = self.processor.export_reference()
            msg.baseline_adc = profile.baseline_adc
            msg.activation_threshold_adc = profile.activation_threshold_adc
            msg.valid_mask = profile.valid_mask
        self.reference_publisher.publish(msg)
        self._published_reference = key
