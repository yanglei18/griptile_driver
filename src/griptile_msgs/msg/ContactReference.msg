# Authority for the displayed contact image. Coefficients are present only in phase "ready".
std_msgs/Header header
string sensor_id
string session_id
uint32 stream_epoch
string hardware_id
string phase
float32 progress
string reason
float64 span_adc
string direction
float64[<=400] baseline_adc
float64[<=400] activation_threshold_adc
bool[<=400] valid_mask
