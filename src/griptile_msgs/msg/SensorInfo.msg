# Stream interpretation and selected serial binding; recorded with samples.
# Header stamp describes metadata publication. Acquisition clocks live in samples.
std_msgs/Header header
uint16 schema_version
string sensor_id
string session_id
uint32 stream_epoch
bool connected
string port
# Empty when the active board cannot be identified independently of its port.
string hardware_id
string protocol_id
string layout_id
uint16 rows
uint16 cols
string unit
string clock_domain
string timing_source
string time_description
bool has_hardware_timestamp
bool has_hardware_sequence
bool has_documented_checksum
string driver_version
