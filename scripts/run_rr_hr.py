# cd /home/pi/Desktop
# python3 final.py -u

from copy import copy
from enum import Enum

import numpy as np
from numpy import pi
from scipy.signal import butter, sosfilt, firwin, filtfilt
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from time import sleep, mktime
import subprocess
from datetime import datetime
import pyrebase

from acconeer.exptool import configs, utils
from acconeer.exptool.clients import SocketClient, SPIClient, UARTClient
from acconeer.exptool.pg_process import PGProccessDiedException, PGProcess
from acconeer.exptool.structs import configbase

# Envelope service constants

PEAK_MERGE_LIMIT_M = 0.005
TARGET_RANGE_CM = [26, 56]
RANGE_CALIBRATION_WAIT_TIME_SEC = 5
CONT_REALTIME_MONITORING = False

# IQ service constants

RADAR_FS = 40
WINDOW_LENGTH = 14
START_DISTANCE = 0.3
STOP_DISTANCE = 0.6
MIN_DISTANCE = 0.2
MAX_DISTANCE = 1.0
MAX_SAMPLING_TIME = 5 * 60
DEBUG_EXEC = False
DEBUG = False
DEBUG_IQ_BUFFER = []
DEBUG_HR_BUFFER = []
DEBUG_PLOT = False

firebaseConfig = {
  "apiKey": "AIzaSyATjdo9-LnDVRgxfeuQaOopO4EIFmTXEik",
  "authDomain": "bioradar.firebaseapp.com",
  "databaseURL": "https://bioradar-default-rtdb.europe-west1.firebasedatabase.app",
  "projectId": "bioradar",
  "storageBucket": "bioradar.appspot.com",
  "messagingSenderId": "857754767607",
  "appId": "1:857754767607:web:78f269b3f4da42c60343f2",
  "measurementId": "G-0SC2CY2RDM"
}

if DEBUG_PLOT:
    plt.ion()  # To update matplotlib plots in real-time

def main():
    ## -- Connect to Firebase -- ##

    print("Connecting to Google Firebase...\n")
    firebase = pyrebase.initialize_app(firebaseConfig)
    db = firebase.database()
    for_js = 0
    data = {"BR": 0, "HR": 0, 'HRV': 0}
    
    ## -- Start envelope service to perform range calibration -- ##

    print("Welcome! Connecting to XM112/XB112...")
    
    args = utils.ExampleArgumentParser(num_sens=1).parse_args()
    utils.config_logging(args)

    if args.socket_addr:
        client = SocketClient(args.socket_addr)
    elif args.spi:
        client = SPIClient()
    else:
        port = args.serial_port or utils.autodetect_serial_port()
        client = UARTClient(port)

    sensor_config = get_sensor_config_envelope()
    processing_config = get_processing_config_envelope()
    sensor_config.sensor = args.sensors
    try:
        session_info = client.setup_session(sensor_config)
    except:
        print("ERROR: Could not connect to XM112/XB112 module. Connect and try again.\n")
        process = subprocess.run('python3 gui_restart.py', shell=True)
        exit()
        
    print("Starting range calibration.\nPress Ctrl-C to end session...\n")
    client.start_session()
    interrupt_handler = utils.ExampleInterruptHandler()
    processor = EnvelopeProcessor(sensor_config, processing_config, session_info)
    
    x = 0
    dist = 0
    
    presence = "N/A"
    presence_counter = 0
    within_range = False
    within_range_text = "No"
    start_counter = False
    range_counter = RANGE_CALIBRATION_WAIT_TIME_SEC

    while not interrupt_handler.got_signal:
        info, sweep = client.get_next()
        
        data, r = processor.process(sweep)  

        if data["found_peaks"] is not None:
            peaks = np.take(r,data["found_peaks"]) * 100.0
            if data["found_peaks"]:
                dist = peaks[0]
                presence_counter += 1

        if x == RADAR_FS/2:
            print("\033c")  # Clear terminal
            if presence_counter == 0 and presence is not "Not detected":
                presence = "N/A"
                within_range_text = "No"
                within_range = False
                start_counter = False
                if range_counter is not RANGE_CALIBRATION_WAIT_TIME_SEC:
                    range_counter = RANGE_CALIBRATION_WAIT_TIME_SEC
                dist = 0
            elif presence_counter > 0:
                presence = "Yes"
                if dist > TARGET_RANGE_CM[-1]:
                    within_range_text = "Too far away!"
                    within_range = False
                    start_counter = False
                    if range_counter is not RANGE_CALIBRATION_WAIT_TIME_SEC:
                        range_counter = RANGE_CALIBRATION_WAIT_TIME_SEC
                elif dist < TARGET_RANGE_CM[0]:
                    within_range_text = "Too close!"
                    within_range = False
                    start_counter = False
                    if range_counter is not RANGE_CALIBRATION_WAIT_TIME_SEC:
                        range_counter = RANGE_CALIBRATION_WAIT_TIME_SEC
                else:
                    start_counter = True
                    within_range_text = "Yes. Don't speak or move!"
                    within_range = True
                    start_counter = True
                    #print("")
                presence_counter = 0
            print("Performing range calibration\nStay still within 26-56 cm from sensor.\n")
            print("Distance: %.1f cm\nPresence: %s\nWithin range: %s" % (dist, presence, within_range_text))
            if not start_counter:
                x = 0
        
        if start_counter:
            if range_counter <= 0 and x == RADAR_FS/2:
                print("\nRange calibration complete! Starting...\n")
                break
            else:
                if x == RADAR_FS/2:
                    print("\n--> Starting in %i... <--" % range_counter)
                if x == RADAR_FS:
                    range_counter -= 1
                    
            if x == RADAR_FS:
                x = 0    
            
        x += 1
        
    START_DISTANCE = round((dist-6)/100, 2)
    STOP_DISTANCE = round((dist+6)/100, 2)
        
    client.disconnect()
    
    # Switch to IQ service
    
    sensor_config = get_sensor_config_iq()
    processing_config = ProcessingConfigurationIQ()
    sensor_config.sensor = args.sensors
    
    try:
        session_info = client.setup_session(sensor_config)
    except:
        print("ERROR: Could not switch to IQ service.")
        exit()
        
    processor = VitalsProcessor(sensor_config, processing_config, session_info)
    client.start_session()

    interrupt_handler = utils.ExampleInterruptHandler()
    print("Target range: %.2f m" % dist) # DEBUG
    print("Calibrated range interval: %.2f to %.2f m\n" % (START_DISTANCE, STOP_DISTANCE))
    
    print("Starting. Press Ctrl-C to end session...\n")

    x = RADAR_FS
    #begin_t = datetime.now()
    end_t = datetime.now()
    load_bar_str = "["
    load_count = 0

    while not interrupt_handler.got_signal:
        info, data = client.get_next()

        if (x <= 0):
            begin_t = datetime.now()

        rr, hr, hrv, dist, rbm = processor.process(data)

        if (x <= 0):
            print("\033c")  # Clear terminal
            print("Range interval: %.2f to %.2f m\n" % (START_DISTANCE, STOP_DISTANCE))
            
            if load_count < WINDOW_LENGTH:
                load_count += 1
            end_t = datetime.now()
            te = (end_t - begin_t).total_seconds() * 1000
            if DEBUG_EXEC:
                print("[", end_t.strftime('%d:%m:%Y %H:%M:%S'), "]\t\tExecution time: %.2f ms" % te)
            
            for l in range(0, load_count):
                load_bar_str += "*"
        
            for l in range(0, WINDOW_LENGTH-load_count):
                load_bar_str += "-"
            
            load_bar_str += "] (" + str(round(load_count/WINDOW_LENGTH * 100)) + "%)"

            print("Acquiring RR and HR.\nStay still for %s seconds.\n" % str(WINDOW_LENGTH))
            print(load_bar_str)
            if not rbm:
                print("No RBM detected.")
            else:
                print("RBM detected! Readings may be incorrect.")
            load_bar_str = "["
            
            
            if load_count >= WINDOW_LENGTH:
                print("")
            
            if hr is not 0:
                print("> Respiration rate: %.2f breaths/min" % rr)
                print("> Heart rate: %.2f beats/min" % hr)
                #print("> Heart rate variability: %.2f milliseconds" % hrv)
                #print("\t\t\t\tDistance from sensor: %.2f m" % dist)
                
                data = {"BR": rr, "HR": hr, 'HRV': hrv}
                for_js = int(mktime(end_t.timetuple())) * 1000
                db.child('Users').child('Zain').child(for_js).set(data)
                
            print("")
                
            x = RADAR_FS

        x -= 1
        
        if not CONT_REALTIME_MONITORING and load_count >= WINDOW_LENGTH:
            break

        if DEBUG:
            DEBUG_IQ_BUFFER.append(data)

    print("Disconnecting...\nThis window will close in 10 seconds.")
    client.disconnect()
    sleep(10)
    subprocess.run('python3 scripts/GUI_restart.py', shell=True)
    exit()


def get_sensor_config_envelope():
    config = configs.EnvelopeServiceConfig()
    config.range_interval = [0.2, 0.8]
    config.update_rate = RADAR_FS
    config.gain = 0.5
    config.running_average_factor = 0  # Use averaging in detector instead of in API

    return config


class EnvelopeProcessor:
    def __init__(self, sensor_config, processing_config, session_info):
        self.session_info = session_info

        self.f = sensor_config.update_rate

        num_depths = self.session_info["data_length"]

        self.current_mean_sweep = np.zeros(num_depths)
        self.last_mean_sweep = np.full(num_depths, np.nan)
        self.sweeps_since_mean = 0

        self.sc_sum_bg_sweeps = np.zeros(num_depths)
        self.sc_sum_squared_bg_sweeps = np.zeros(num_depths)
        self.sc_bg_sweep_mean = np.full(num_depths, np.nan)
        self.sc_bg_sweep_std = np.full(num_depths, np.nan)
        self.sc_bg_threshold = np.full(num_depths, np.nan)
        self.sc_used_mean = np.full(num_depths, np.nan)
        self.sc_used_std = np.full(num_depths, np.nan)
        self.sc_used_threshold = np.full(num_depths, np.nan)

        self.sc_bg_calculated = False

        self.history_length_s = processing_config.history_length_s
        self.main_peak_hist_sweep_idx = []
        self.main_peak_hist_dist = []
        self.minor_peaks_hist_sweep_idx = []
        self.minor_peaks_hist_dist = []
        self.above_thres_hist_sweep_idx = []
        self.above_thres_hist_dist = []

        self.r = utils.get_range_depths(sensor_config, session_info)
        self.dr = self.r[1] - self.r[0]
        self.sweep_index = 0

        self.update_processing_config(processing_config)

    def update_processing_config(self, processing_config):
        self.nbr_average = processing_config.nbr_average
        self.threshold_type = processing_config.threshold_type
        self.peak_sorting_method = processing_config.peak_sorting_type

        self.fixed_threshold_level = processing_config.fixed_threshold

        self.sc_sensitivity = processing_config.sc_sensitivity
        self.sc_bg_nbr_sweeps = processing_config.sc_nbr_sweep_for_bg
        self.sc_load_save_bg = processing_config.sc_load_save_bg

        self.idx_cfar_pts = np.round((
            processing_config.cfar_guard_cm / 100.0 / 2.0 / self.dr
            + np.arange(processing_config.cfar_window_cm / 100.0 / self.dr)))

        self.cfar_one_sided = processing_config.cfar_one_sided
        self.cfar_sensitivity = processing_config.cfar_sensitivity

        self.update_sc_threshold()

        self.history_length_s = processing_config.history_length_s

    def update_sc_threshold(self):
        self.sc_used_threshold = self.sc_used_mean + \
            (1.0/(self.sc_sensitivity + 1e-10) - 1.0) * self.sc_used_std

    def get_sc_threshold(self, sweep):
        # Collect first sweeps to construct a stationary clutter threshold
        # Accumulate sweeps instead of saving each for lower memory footprint
        if self.sweep_index < self.sc_bg_nbr_sweeps:
            self.sc_sum_bg_sweeps += sweep
            self.sc_sum_squared_bg_sweeps += np.square(sweep)

        if self.sweep_index >= self.sc_bg_nbr_sweeps - 1 and not self.sc_bg_calculated:
            self.sc_bg_sweep_mean = self.sc_sum_bg_sweeps / self.sc_bg_nbr_sweeps
            mean_square = self.sc_sum_squared_bg_sweeps / self.sc_bg_nbr_sweeps
            square_mean = np.square(self.sc_bg_sweep_mean)
            self.sc_bg_sweep_std = np.sqrt((mean_square - square_mean)
                                           * self.sc_bg_nbr_sweeps / (self.sc_bg_nbr_sweeps - 1))

            self.sc_bg_calculated = True

            self.sc_load_save_bg.buffered_data = np.array([
                self.sc_bg_sweep_mean,
                self.sc_bg_sweep_std,
            ])
            # self.sc_load_save_bg.loaded_data = self.sc_load_save_bg.buffered_data

        # Checking if user loaded a threshold and if it is compatible
        if self.sc_load_save_bg.error is None:
            loaded_threshold_data = self.sc_load_save_bg.loaded_data

            if loaded_threshold_data is not None:
                try:
                    if not isinstance(loaded_threshold_data, np.ndarray):
                        self.sc_load_save_bg.error = "Wrong type"
                    elif np.iscomplexobj(loaded_threshold_data):
                        self.sc_load_save_bg.error = "Wrong type (is complex)"
                    elif loaded_threshold_data.shape != (2, sweep.size):
                        self.sc_load_save_bg.error = "Size mismatch"
                    else:
                        self.sc_used_mean = loaded_threshold_data[0, :]
                        self.sc_used_std = loaded_threshold_data[1, :]
                except Exception:
                    self.sc_used_mean = np.full(sweep.shape, np.nan)
                    self.sc_used_std = np.full(sweep.shape, np.nan)
                    self.sc_load_save_bg.error = "Invalid threshold data"
            else:
                self.sc_used_mean = np.full(sweep.shape, np.nan)
                self.sc_used_std = np.full(sweep.shape, np.nan)

        self.update_sc_threshold()

    def calculate_cfar_threshold(self, sweep, idx_cfar_pts, alpha, one_side):

        threshold = np.full(sweep.shape, np.nan)

        start_idx = np.max(idx_cfar_pts)
        if one_side:
            rel_indexes = -idx_cfar_pts
            end_idx = sweep.size
        else:
            rel_indexes = np.concatenate((-idx_cfar_pts, +idx_cfar_pts), axis=0)
            end_idx = sweep.size - start_idx

        for idx in np.arange(start_idx, end_idx):
            threshold[int(idx)] = 1.0 / (alpha + 1e-10) \
                * np.mean(sweep[(idx + rel_indexes).astype(int)])

        return threshold

    def find_first_point_above_threshold(self, sweep, threshold):

        if threshold is None or np.all(np.isnan(threshold)):
            return None

        points_above = (sweep > threshold)

        if not np.any(points_above):
            return None

        return np.argmax(points_above)

    def find_peaks(self, sweep, threshold):
        #  Not written for optimal speed.

        if threshold is None or np.all(np.isnan(threshold)):
            return []

        found_peaks = []

        # Note: at least 3 samples above threshold are required to form a peak

        d = 1
        N = len(sweep)
        while d < (N - 1):
            # Skip to when threshold starts, applicable only for CFAR
            if np.isnan(threshold[d - 1]):
                d += 1
                continue

            # Break when threshold ends, applicable only for CFAR
            if np.isnan(threshold[d + 1]):
                break

            # At this point, threshold is defined (not Nan)

            # If the current point is not over threshold, the next will not be a peak
            if sweep[d] <= threshold[d]:
                d += 2
                continue

            # Continue if previous point is not over threshold
            if sweep[d - 1] <= threshold[d - 1]:
                d += 1
                continue

            # Continue if this point isn't larger than the previous
            if sweep[d - 1] >= sweep[d]:
                d += 1
                continue

            # A peak is either a single point or a plateau consisting of several equal points,
            # all over their threshold. The closest neighboring points on each side of the
            # point/plateau must have a lower value and be over their threshold.
            # Now, decide if the following point(s) are a peak:

            d_upper = d + 1
            while True:
                if (d_upper) >= (N - 1):  # If out of range or on last point
                    break

                if np.isnan(threshold[d_upper]):
                    break

                if sweep[d_upper] <= threshold[d_upper]:
                    break

                if sweep[d_upper] > sweep[d]:
                    break
                elif sweep[d_upper] < sweep[d]:
                    delta = d_upper - d
                    found_peaks.append(d + int(np.ceil((delta - 1) / 2.0)))
                    break
                else:  # equal
                    d_upper += 1

            d = d_upper

        return found_peaks

    def merge_peaks(self, peak_indexes, merge_max_range):
        merged_peaks = copy(peak_indexes)

        while True:
            num_neighbors = np.zeros(len(merged_peaks))  # number of neighbors
            for i, p in enumerate(merged_peaks):
                num_neighbors[i] = np.sum(np.abs(np.array(merged_peaks) - p) < merge_max_range)

            # First peak with max number of neighbors
            i_peak = np.argmax(num_neighbors)  # returns arg of first max

            if num_neighbors[i_peak] <= 1:
                break

            peak = merged_peaks[i_peak]

            remove_mask = np.abs(np.array(merged_peaks) - peak) < merge_max_range
            peaks_to_remove = np.array(merged_peaks)[remove_mask]

            for p in peaks_to_remove:
                merged_peaks.remove(p)

            # Add back mean peak
            merged_peaks.append(int(round(np.mean(peaks_to_remove))))

            merged_peaks.sort()

        return merged_peaks

    def sort_peaks(self, peak_indexes, sweep):
        amp = np.array([sweep[int(i)] for i in peak_indexes])
        r = np.array([self.r[int(i)] for i in peak_indexes])

        PeakSorting = ProcessingConfigurationEnvelope.PeakSorting
        if self.peak_sorting_method == PeakSorting.CLOSEST:
            quantity_to_sort = r
        elif self.peak_sorting_method == PeakSorting.STRONGEST:
            quantity_to_sort = -amp
        elif self.peak_sorting_method == PeakSorting.STRONGEST_REFLECTOR:
            quantity_to_sort = -amp * r ** 2
        elif self.peak_sorting_method == PeakSorting.STRONGEST_FLAT_REFLECTOR:
            quantity_to_sort = -amp * r
        else:
            raise Exception("Unknown peak sorting method")

        return [peak_indexes[i] for i in quantity_to_sort.argsort()]

    def process(self, sweep):
        # Accumulate sweeps for stationary clutter threshold and check if user has
        # loaded one from disk
        self.get_sc_threshold(sweep)

        # Average envelope sweeps, written to handle varying nbr_average
        weight = 1.0/(1.0 + self.sweeps_since_mean)
        self.current_mean_sweep = weight*sweep + (1.0 - weight)*self.current_mean_sweep
        self.sweeps_since_mean += 1

        # Determining threshold
        if self.threshold_type is ProcessingConfigurationEnvelope.ThresholdType.FIXED:
            threshold = self.fixed_threshold_level * np.ones(sweep.size)
        elif self.threshold_type is ProcessingConfigurationEnvelope.ThresholdType.RECORDED:
            threshold = self.sc_used_threshold
        elif self.threshold_type is ProcessingConfigurationEnvelope.ThresholdType.CFAR:
            threshold = self.calculate_cfar_threshold(
                self.current_mean_sweep,
                self.idx_cfar_pts,
                self.cfar_sensitivity,
                self.cfar_one_sided,
            )
        else:
            print("Unknown thresholding method")

        found_peaks = None

        # If a new averaged sweep is ready for processing
        if self.sweeps_since_mean >= self.nbr_average:
            self.sweeps_since_mean = 0
            self.last_mean_sweep = self.current_mean_sweep.copy()
            self.current_mean_sweep *= 0

            # Find the first delay over threshold. Used in tank-level when monitoring changes
            # in the direct leakage.
            first_point_above_threshold = self.find_first_point_above_threshold(
                self.last_mean_sweep, threshold)

            # First peak-finding, then peak-merging, finallay peak sorting.
            found_peaks = self.find_peaks(self.last_mean_sweep, threshold)
            if len(found_peaks) > 1:
                found_peaks = self.merge_peaks(found_peaks, np.round(PEAK_MERGE_LIMIT_M/self.dr))
                found_peaks = self.sort_peaks(found_peaks, self.last_mean_sweep)

            # Adding main peak to history
            if len(found_peaks) > 0:
                self.main_peak_hist_sweep_idx.append(self.sweep_index)
                self.main_peak_hist_dist.append(self.r[found_peaks[0]])

            # Adding minor peaks to history
            for i in range(1, len(found_peaks)):
                self.minor_peaks_hist_sweep_idx.append(self.sweep_index)
                self.minor_peaks_hist_dist.append(self.r[found_peaks[i]])

            # Adding first distance above threshold to history
            if first_point_above_threshold is not None:
                self.above_thres_hist_sweep_idx.append(self.sweep_index)
                self.above_thres_hist_dist.append(self.r[first_point_above_threshold])

            # Removing old main peaks from history
            while len(self.main_peak_hist_sweep_idx) > 0 and \
                    (self.sweep_index - self.main_peak_hist_sweep_idx[0]) \
                    > self.history_length_s*self.f:
                self.main_peak_hist_sweep_idx.pop(0)
                self.main_peak_hist_dist.pop(0)

            # Removing old minor peaks from history
            while len(self.minor_peaks_hist_sweep_idx) > 0 and \
                    (self.sweep_index - self.minor_peaks_hist_sweep_idx[0]) \
                    > self.history_length_s*self.f:
                self.minor_peaks_hist_sweep_idx.pop(0)
                self.minor_peaks_hist_dist.pop(0)

            # Removing old first distance above threshold from history
            while len(self.above_thres_hist_sweep_idx) > 0 and \
                    (self.sweep_index - self.above_thres_hist_sweep_idx[0]) \
                    > self.history_length_s*self.f:
                self.above_thres_hist_sweep_idx.pop(0)
                self.above_thres_hist_dist.pop(0)

        out_data = {
            "sweep": sweep,
            "last_mean_sweep": self.last_mean_sweep,
            "threshold": threshold,
            "main_peak_hist_sweep_s": (np.array(self.main_peak_hist_sweep_idx)
                                       - self.sweep_index)/self.f,
            "main_peak_hist_dist": np.array(self.main_peak_hist_dist),
            "minor_peaks_hist_sweep_s": (np.array(self.minor_peaks_hist_sweep_idx)
                                         - self.sweep_index)/self.f,
            "minor_peaks_hist_dist": np.array(self.minor_peaks_hist_dist),
            "above_thres_hist_sweep_s": (np.array(self.above_thres_hist_sweep_idx)
                                         - self.sweep_index)/self.f,
            "above_thres_hist_dist": np.array(self.above_thres_hist_dist),
            "sweep_index": self.sweep_index,
            "found_peaks": found_peaks,
        }

        self.sweep_index += 1

        return out_data, self.r


class ProcessingConfigurationEnvelope(configbase.ProcessingConfig):
    class ThresholdType(Enum):
        FIXED = "Fixed"
        RECORDED = "Recorded"
        CFAR = "CFAR"

    class PeakSorting(Enum):
        STRONGEST = "Strongest signal"
        CLOSEST = "Closest signal"
        STRONGEST_REFLECTOR = "Strongest reflector"
        STRONGEST_FLAT_REFLECTOR = "Strongest flat reflector"

    VERSION = 1

    nbr_average = configbase.FloatParameter(
        label="Sweep averaging",
        default_value=5,
        limits=(1, 100),
        logscale=True,
        decimals=0,
        updateable=True,
        order=0,
        visible=True,
        help=(
            "The number of envelope sweeps to be average into one then used for"
            " distance detection."
        ),
    )

    threshold_type = configbase.EnumParameter(
        label="Threshold type",
        default_value=ThresholdType.FIXED,
        enum=ThresholdType,
        updateable=True,
        order=5,
        help="Setting the type of threshold",
    )

    fixed_threshold = configbase.FloatParameter(
        label="Fixed threshold level",
        default_value=800,
        limits=(1, 20000),
        decimals=0,
        updateable=True,
        order=10,
        visible=lambda conf: conf.threshold_type == conf.ThresholdType.FIXED,
        help=(
            "Sets the value of fixed threshold. The threshold has this constant value over"
            " the full sweep."
        ),
    )

    sc_nbr_sweep_for_bg = configbase.FloatParameter(
        label="Number of sweeps for background estimation",
        default_value=20,
        limits=(2, 200),
        decimals=0,
        visible=lambda conf: conf.threshold_type == conf.ThresholdType.RECORDED,
        updateable=True,
        order=20,
        help=(
            "The number of (non-averaged) sweeps collected for calculating the Stationary"
            " Clutter threshold."
        ),
    )

    sc_load_save_bg = configbase.ReferenceDataParameter(
        label="Recorded threshold",
        visible=lambda conf: conf.threshold_type == conf.ThresholdType.RECORDED,
        order=23,
        help=(
            "Load/Save a recorded threshold from/to disk."
        ),
    )

    sc_sensitivity = configbase.FloatParameter(
        label="Stationary clutter sensitivity",
        default_value=0.3,
        limits=(0.01, 1),
        logscale=True,
        visible=lambda conf: conf.threshold_type == conf.ThresholdType.RECORDED,
        decimals=4,
        updateable=True,
        order=24,
        help=(
            "Value between 0 and 1 that sets the threshold. A low sensitivity will set a "
            "high threshold, resulting in only few false alarms but might result in "
            "missed detections."
        ),
    )

    cfar_sensitivity = configbase.FloatParameter(
        label="CFAR sensitivity",
        default_value=0.5,
        limits=(0.01, 1),
        logscale=True,
        visible=lambda conf: conf.threshold_type == conf.ThresholdType.CFAR,
        decimals=4,
        updateable=True,
        order=40,
        help=(
            "Value between 0 and 1 that sets the threshold. A low sensitivity will set a "
            "high threshold, resulting in only few false alarms but might result in "
            "missed detections."
        ),
    )

    cfar_guard_cm = configbase.FloatParameter(
        label="CFAR guard",
        default_value=12,
        limits=(1, 20),
        unit="cm",
        decimals=1,
        visible=lambda conf: conf.threshold_type == conf.ThresholdType.CFAR,
        updateable=True,
        order=41,
        help=(
            "Range around the distance of interest that is omitted when calculating "
            "CFAR threshold. Can be low, ~4 cm, for Profile 1, and should be "
            "increased for higher Profiles."
        ),
    )

    cfar_window_cm = configbase.FloatParameter(
        label="CFAR window",
        default_value=3,
        limits=(0.1, 20),
        unit="cm",
        decimals=1,
        visible=lambda conf: conf.threshold_type == conf.ThresholdType.CFAR,
        updateable=True,
        order=42,
        help=(
            "Range next to the CFAR guard from which the threshold level will be calculated."
        ),
    )

    cfar_one_sided = configbase.BoolParameter(
        label="Use only lower distance to set threshold",
        default_value=False,
        visible=lambda conf: conf.threshold_type == conf.ThresholdType.CFAR,
        updateable=True,
        order=43,
        help=(
            "Instead of determining the CFAR threshold from sweep amplitudes from "
            "distances both closer and a farther, use only closer. Helpful e.g. for "
            "fluid level in small tanks, where many multipath signal can apprear "
            "just after the main peak."
        ),
    )

    peak_sorting_type = configbase.EnumParameter(
        label="Peak sorting",
        default_value=PeakSorting.STRONGEST,
        enum=PeakSorting,
        updateable=True,
        order=100,
        help="Setting the type of peak sorting method.",
    )

    history_length_s = configbase.FloatParameter(
        default_value=10,
        limits=(3, 1000),
        updateable=True,
        logscale=True,
        unit="s",
        label="History length",
        order=198,
        help="Length of time history for plotting."
    )

    show_first_above_threshold = configbase.BoolParameter(
        label="Show first distance above threshold",
        default_value=False,
        updateable=True,
        order=199,
        help=(
            "When detect in the presence of object very close to the sensor, the "
            "strong direct leakage might cause that no well shaped peaks are detected, "
            "even though the envelope signal is above the threshold. Therefore the "
            "first distace where the signal is above the threshold can be used as an "
            "alternative to peak detection."
        ),
    )

    def check_sensor_config(self, sensor_config):
        alerts = []

        if sensor_config.update_rate is None:
            alerts.append(configbase.Error("update_rate", "Must be set"))

        if not sensor_config.noise_level_normalization:
            if self.threshold_type == self.ThresholdType.FIXED:
                alerts.append(configbase.Warning(
                    "noise_level_normalization",
                    (
                        "Enabling noise level normalization is "
                        "recommended with Fixed threshold"
                    )
                ))

        return alerts


get_processing_config_envelope = ProcessingConfigurationEnvelope

def get_sensor_config_iq():
    config = configs.IQServiceConfig()
    
    config.range_interval = [START_DISTANCE, STOP_DISTANCE]  # Minimum: 0.2, Maximum: 1.0
    config.update_rate = RADAR_FS
    config.gain = 0.5
    config.repetition_mode = configs.IQServiceConfig.RepetitionMode.SENSOR_DRIVEN
    return config

class ProcessingConfigurationIQ(configbase.ProcessingConfig):
    VERSION = 1

    hist_plot_len = configbase.FloatParameter(
        label="Plot length",
        unit="s",
        default_value=10,
        limits=(1, 30),
        decimals=0,
    )

class VitalsProcessor:
    def __init__(self, sensor_config, processing_config, session_info):
        self.config = sensor_config
        assert sensor_config.update_rate is not None
        
        self.rr_window_length = 10
        self.hr_window_length = WINDOW_LENGTH
        self.update_interval = 2
        self.min_prominence = 0.03
        self.ibi_factor = 0.707
        self.exp_avg_alpha = 0.3
        self.h_filter_order = 300
        self.h_filter_fc = [1, 3]
        self.h_filter_beta = 1
        self.radar_freq = sensor_config.update_rate
        
        self.sweep_index = 0
        self.hr_window_sweeps = self.hr_window_length * self.radar_freq
        self.rr_window_sweeps = self.rr_window_length * self.radar_freq
        self.update_interval_sweeps = self.update_interval * self.radar_freq

        # Signal buffers

        self.signal_buffer = np.zeros(int(self.hr_window_sweeps), dtype="float")
        self.r_buffer = np.zeros(int(self.rr_window_sweeps), dtype="float")
        self.h_buffer = np.zeros(int(self.hr_window_sweeps), dtype="float")
        self.hr_buffer = np.zeros(5, dtype="float")
        self.buffer_filled = False

        # Filter characteristics

        self.r_sos = np.concatenate(butter(2, 2 * 0.3 / self.radar_freq))
        self.r_zi = np.zeros((1, 2))
        
        self.r_b = firwin(10, 0.5, window=('kaiser', 1), fs=self.radar_freq)
        self.h_b = firwin(self.h_filter_order, self.h_filter_fc, window=('kaiser', self.h_filter_beta), fs=self.radar_freq, pass_zero=False)
        
        # Latest HR, RR and HRV readings
        
        self.hr = 0
        self.rr = 0
        self.hrv = 0

        # Parameters for exponential weighted averaging

        self.ewa_enable = True
        self.phase_weights_alpha = 0.9
        self.peak_loc_alpha = 0.95
        self.sweep_alpha = 0.7
        self.amp_alpha = 0.95
        self.last_avg_sweep = 0
        self.avg_amp = 0
        self.avg_phase_weights = 0
        self.avg_sweep = 0
        self.avg_peak_loc = 0
        self.acc_displacement = 0

        self.begin_t = 0
        self.end_t = 0
        
        # For find_dist_from_sensor()
        
        self.dist_array = np.array([])
        self.dist = 0
        self.first_sample = True
        
        self.avg_mean_dev = 500
        self.iterations = 0
        self.hr_rbm = 0
        self.avg_hr_value = 0

    def process(self, sweep):
        self.acc_displacement += self.phase_tracking(sweep)
        #r_signal, self.r_zi = sosfilt(self.r_sos, np.array([self.acc_displacement+self.r_buffer[-1]]), zi=self.r_zi)
        #r_signal = np.append(r_signal, self.r_buffer)

        if self.buffer_filled:
            # Implement FILO
            # self.signal_buffer[0:-2] = self.signal_buffer[1:-1]
            # self.signal_buffer[-1] = self.acc_displacement
            self.push(self.acc_displacement, self.signal_buffer)

            # Once raw signal buffer is filled, filter out heartbeat signal after each update interval
            if self.sweep_index % self.update_interval_sweeps == 0:
                #print(len(self.signal_buffer))
                #self.begin_t = datetime.now()
                self.h_buffer = filtfilt(self.h_b, 1, self.signal_buffer, axis=0, padlen=len(self.signal_buffer)-1)
                try:
                    self.hr = self.smart_peak_detector()
                except:
                    print("Encountered an error in HR estimation.\nClosing window in 10 seconds.")
                    sleep(10)
                
                #self.dist = self.get_dist_from_sensor(sweep)

                #self.end_t = datetime.now()
                #te = (self.end_t - self.begin_t).total_seconds() * 1000
                #print("Execution time: ", te)

                if DEBUG:
                    DEBUG_HR_BUFFER.append(self.hr)
                    
                if DEBUG_PLOT:
                    plt.clf()
                    plt.plot(self.h_buffer)
                    plt.plot(self.signal_buffer)
                    plt.pause(0.0001)
                    plt.show()
                #print("Heart rate: %i bpm" % hr)
                #self.push(self.hr_buffer, hr)
                #self.hr_buffer = np.append(self.hr_buffer, hr)

        else:
            self.signal_buffer[self.sweep_index] = self.acc_displacement
            
        #self.r_buffer = filtfilt(self.r_b, 1, self.signal_buffer, padlen=len(self.r_buffer)-1)
        self.r_buffer, self.r_zi = sosfilt(self.r_sos, self.signal_buffer, zi=self.r_zi)
        self.find_rr()

        self.sweep_index += 1

        if self.sweep_index >= self.hr_window_sweeps:
            if not self.buffer_filled:
                self.buffer_filled = True
            self.sweep_index = 0

        return self.rr, self.hr, self.hrv, self.dist, self.rbm_detection()
    
    def get_dist_from_sensor(self, sweep):
        d = 0
        iq_amp = np.abs(sweep)
        if self.dist_array.size == 0:
            self.dist_array = np.linspace(START_DISTANCE, STOP_DISTANCE, len(iq_amp))
            
        #p, _ = find_peaks(iq_amp)
        
        p = np.argmax(iq_amp)
        
        if self.first_sample:
            d = self.dist_array[p]
            self.first_sample = False
        else:
            d = self.exp_ma(self.dist_array[p], self.dist, 0.3)
        
#         if p.size > 0:
#             p = np.argmax(iq_amp)
#             
#             if self.first_sample:
#                 d = self.dist_array[p]
#                 self.first_sample = False
#             else:
#                 d = self.exp_ma(self.dist_array[p], self.dist, 0.99)
        
        return d

    def phase_tracking(self, sweep):
        if self.sweep_index == 0:
            self.avg_sweep = np.array(sweep)
            self.avg_amp = np.abs(sweep)
            self.avg_peak_loc = np.argmax(self.avg_amp)
            delta_dist = 0

        else:
            self.avg_sweep = self.exp_ma(sweep, self.avg_sweep, self.sweep_alpha)
            amp = np.abs(self.avg_sweep)
            self.avg_amp = self.exp_ma(amp, self.avg_amp, self.amp_alpha)
            peak_loc = np.argmax(self.avg_amp)
            self.avg_peak_loc = self.exp_ma(peak_loc, self.avg_peak_loc, self.peak_loc_alpha)

            # peak_index = int(round(self.avg_peak_loc))
            # peak = np.mean(self.avg_sweep[peak_idx - 50: peak_idx + 50])
            # self.push(peak, self.peak_history)

            delta_phase = self.avg_sweep * np.conj(self.last_avg_sweep)

            phase_weights = np.imag(delta_phase)
            if self.avg_phase_weights is None:
                self.avg_phase_weights = phase_weights
            else:
                self.avg_phase_weights = self.exp_ma(phase_weights, self.avg_phase_weights, self.phase_weights_alpha)

            weights = np.abs(self.avg_phase_weights) * amp

            delta_dist = np.dot(weights, np.angle(delta_phase))
            delta_dist *= 2.5 / (2.0 * pi * sum(weights + 0.00001))

            self.last_avg_sweep = self.avg_sweep

        return delta_dist

    def exp_ma(self, new, prev, alpha):
        return alpha * prev + (1 - alpha) * new

    def push(self, sample, buffer):
        res = np.empty_like(buffer)
        res[0] = sample
        res[1:] = buffer[:-1]
        buffer[...] = res

    def smart_peak_detector(self):
        peak_idx, _ = find_peaks(self.h_buffer, prominence=self.min_prominence)
        if len(peak_idx) > 0:
            peak_mag = self.h_buffer[peak_idx]
            peak_mag_sum = np.sum(peak_mag)

            mean_ibi = 0
            num_peaks = len(peak_idx)

            for i in range(1, num_peaks):
                mean_ibi += (peak_idx[i]-peak_idx[i-1])/self.radar_freq * peak_mag[i]/peak_mag_sum

            min_pk_dist = mean_ibi * self.ibi_factor
            if (min_pk_dist*self.radar_freq >= 1):
                peak_idx, _ = find_peaks(self.h_buffer, distance=round(min_pk_dist*self.radar_freq), prominence=self.min_prominence)
            else:
                return -1
             
            num_beats = len(peak_idx)-1
            time_dur = (peak_idx[-1]-peak_idx[0])/self.radar_freq
            
            return round(float(num_beats * 60 / time_dur), 2)
        else:
            return -1

    def find_rr(self):
        maxs = self.find_peaks_custom(self.r_buffer, 100)
        mins = self.find_peaks_custom(-self.r_buffer, 100)
        max_idx = 0
        min_idx = 0
        inhale_time = None
        exhale_time = None
        inhale_dist = 0
        exhale_dist = 0
        exhale = None
        first_peak = None
        while not (inhale_time and exhale_time):
            if not (min_idx < mins.shape[0] and max_idx < maxs.shape[0]):
                break

            if maxs[max_idx, 0] < mins[min_idx, 0]:
                exhale_dist = mins[min_idx, 1] + maxs[max_idx, 1]
                if (exhale_dist > 1 and exhale_dist < 20):
                    exhale_time = mins[min_idx, 0] - maxs[max_idx, 0]
                    if first_peak is None:
                        first_peak = maxs[max_idx, 0]
                    exhale = True
                max_idx += 1
            else:
                inhale_dist = mins[min_idx, 1] + maxs[max_idx, 1]
                if (inhale_dist > 1 and inhale_dist < 20):
                    inhale_time = maxs[max_idx, 0] - mins[min_idx, 0]
                    exhale = False
                    if first_peak is None:
                        first_peak = mins[min_idx, 0]
                min_idx += 1

        breathing = False

        if inhale_time is not None and exhale_time is not None:
            self.rr = round(60.0 / ((inhale_time + exhale_time) / self.radar_freq), 2)
            symmetry = (inhale_dist - exhale_dist) / (inhale_dist + exhale_dist)
            first_peak_rel = first_peak / (inhale_time + exhale_time)
            if 3 < self.rr < 30 and abs(symmetry) < 0.6 and first_peak_rel < 0.7:
                breathing = True

#         if breathing:
#             bstr = "Exhaling" if exhale else "Inhaling"
#             bpm_text = "{}, BPM {:0.1f}, depth {:0.1f} mm".format(bstr, rr, inhale_dist)
#         else:
#             bpm_text = None
            

    def find_peaks_custom(self, env, width):
        n = len(env)
        peaks = np.zeros((0, 2))
        for idx in range(0, n, width):
            mi = np.argmax(env[idx : min(idx + width, n)]) + idx
            mi2 = np.argmax(env[max(mi - width, 0) : min(mi + width, n)])
            mi2 += max(mi - width, 0)
            if mi == mi2 and (0 < mi < n - 1):
                peaks = np.concatenate((peaks, np.array([[mi, env[mi]]])), axis=0)
        return peaks
    
    def rbm_detection(self):
        threshold_factor = 1.5
        mean_deviation = abs(max(self.signal_buffer)-min(self.signal_buffer)) 

        if mean_deviation < (self.avg_mean_dev * threshold_factor):
            self.iterations += 1
            self.avg_mean_dev = (self.avg_mean_dev * self.iterations + mean_deviation) / (self.iterations + 1)
            self.hr_rbm = self.hr
            self.avg_hr_value = self.avg_hr_value * (self.iterations - 1) + self.hr_rbm / self.iterations
            return False
        else:
            if self.avg_hr_value < 50: 
                self.hr_rbm = self.hr
            else:      
                self.hr_rbm = self.hr_rbm * 0.7 + self.avg_hr_value * 0.3
            
            return True

if __name__ == "__main__":
    main()
