import logging
import operator
import signal
import struct
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import serial.tools.list_ports

import acconeer.exptool.structs.configbase as cb
from acconeer.exptool.modes import Mode


#try:
    #import pyqtgraph as pg
#except ImportError:
pg = None

#try:
    #from PyQt5 import QtCore
#except ImportError:
QtCore = None


class ExampleArgumentParser(ArgumentParser):
    def __init__(self, num_sens="+"):
        super().__init__()

        server_group = self.add_mutually_exclusive_group(required=True)
        server_group.add_argument(
            "-u",
            "--uart",
            metavar="port",
            dest="serial_port",
            help="connect via uart (using register-based protocol)",
            nargs="?",
            const="",  # as argparse does not support setting const to None
        )
        server_group.add_argument(
            "-s",
            "--socket",
            metavar="address",
            dest="socket_addr",
            help="connect via socket on given address (using json-based protocol)",
        )
        server_group.add_argument(
            "-spi",
            "--spi",
            dest="spi",
            help="connect via spi (using register-based protocol)",
            action="store_true",
        )

        self.add_argument(
            "--sensor",
            metavar="id",
            dest="sensors",
            type=int,
            default=[1],
            nargs=num_sens,
            help="the sensor(s) to use (default: 1)",
        )

        verbosity_group = self.add_mutually_exclusive_group(required=False)
        verbosity_group.add_argument(
            "-v",
            "--verbose",
            action="store_true",
        )
        verbosity_group.add_argument(
            "-vv",
            "--debug",
            action="store_true",
        )
        verbosity_group.add_argument(
            "-q",
            "--quiet",
            action="store_true",
        )


class ExampleInterruptHandler:
    def __init__(self):
        self._signal_count = 0
        signal.signal(signal.SIGINT, self.interrupt_handler)

    @property
    def got_signal(self):
        return self._signal_count > 0

    def force_signal_interrupt(self):
        self.interrupt_handler(signal.SIGINT, None)

    def interrupt_handler(self, signum, frame):
        self._signal_count += 1
        if self._signal_count >= 3:
            raise KeyboardInterrupt


def mpl_setup_yaxis_for_phase(ax):
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks(np.linspace(-np.pi, np.pi, 5))
    ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"0", r"$\pi/2$", r"$\pi$"])


def timestamp():
    """Get the current date and time formatted as 1999-12-31_23-59-59"""

    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def config_logging(args=None, level=logging.WARN):
    fmt = "{asctime}.{msecs:03.0f} | {levelname:<7} | {processName:<16} | {name:<36} | {message}"
    datefmt = "%H:%M:%S"

    if args is not None:
        if args.debug:
            level = logging.DEBUG
        elif args.verbose:
            level = logging.INFO
        elif args.quiet:
            level = logging.ERROR

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt, datefmt=datefmt, style="{")
    stream_handler.setFormatter(formatter)
    log = logging.getLogger(__name__.split(".")[0])
    log.setLevel(level)
    log.addHandler(stream_handler)

    logging.getLogger(__name__).debug("logging configured")


def set_loglevel(level):
    log = logging.getLogger(__name__.split(".")[0])
    log.setLevel(level)


def autodetect_serial_port():
    port_infos = serial.tools.list_ports.comports()

    for port_info in port_infos:
        port, desc, _ = port_info
        if desc.strip().lower() in ["xb112", "xb122"]:
            print("Autodetected {} on {}\n".format(desc.strip(), port))
            return port

    for port_info in port_infos:
        port, desc, _ = port_info
        if desc == "FT230X Basic UART":
            print("Autodetected FT230X Basic UART on {}\n".format(port))
            return port

    if len(port_infos) == 0:
        print("Could not autodetect serial port, no ports available")
        sys.exit()
    elif len(port_infos) == 1:
        print("Autodetected single available serial port on {}\n".format(port))
        return port_infos[0][0]
    else:
        print("Multiple serial ports are available:")
        for port_info in port_infos:
            port, desc, _ = port_info
            print("  {:<13}  ({})".format(port, desc))
        print("\nRe-run the script with a given port")
        sys.exit()

    print("Could not autodetect serial port")
    sys.exit()


def color_cycler(i=0):
    category10 = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    return category10[i % len(category10)]


def pg_pen_cycler(i=0, style=None, width=2):
    pen = pg.mkPen(color_cycler(i), width=width)
    if style == "--":
        pen.setStyle(QtCore.Qt.DashLine)
    elif style is not None:
        pen.setStyle(style)
    return pen


def pg_brush_cycler(i=0):
    return pg.mkBrush(color_cycler(i))


class SmoothMax:
    def __init__(self, f=None, hysteresis=0.5, tau_decay=2.0, tau_grow=0.5):
        self.fixed_dt = 1 / f if (f is not None and f > 0) else None
        self.hyst = hysteresis
        self.tc_decay = tau_decay
        self.tc_grow = tau_grow

        self.last_t = 0
        self.x = self.y = -1

    def update(self, data):
        m = max(np.nanmax(data), 1e-12)

        if self.fixed_dt is None:
            now = time.time()
            dt = now - self.last_t
            self.last_t = now
        else:
            dt = self.fixed_dt

        ax = np.exp(-dt / self.tc_decay) if self.tc_decay > 1e-3 else 0
        ay = np.exp(-dt / self.tc_grow) if self.tc_grow > 1e-3 else 0

        if m > self.x:
            self.x = m
        else:
            self.x = (1 - ax) * m + ax * self.x

        if self.y < 0:
            self.y = self.x
        elif self.x > self.y:
            self.y = (1 - ay) * self.x * (1 + self.hyst) + ay * self.y
        elif self.x < (1 - self.hyst) * self.y:
            self.y = self.x / (1 - self.hyst)

        return self.y


class SmoothLimits:
    def __init__(self, f=None, hysteresis=0.3, tau_decay=1.5, tau_grow=0.3):
        self.fixed_dt = 1 / f if (f is not None and f > 0) else None
        self.hyst = hysteresis
        self.tc_decay = tau_decay
        self.tc_grow = tau_grow

        self.last_t = 0
        self.x = self.y = self.z = None

    def update(self, data):
        data_lims = (np.nanmin(data), np.nanmax(data))

        if self.fixed_dt is None:
            now = time.time()
            dt = now - self.last_t
            self.last_t = now
        else:
            dt = self.fixed_dt

        if self.x is None:  # First call
            self.x = list(data_lims)
            self.y = list(data_lims)
            self.z = list(data_lims)
            return list(data_lims)

        ad = np.exp(-dt / self.tc_decay) if self.tc_decay > 1e-3 else 0
        ag = np.exp(-dt / self.tc_grow) if self.tc_grow > 1e-3 else 0

        ops = (operator.lt, operator.gt)
        idxs = (0, 1)

        for op, i in zip(ops, idxs):
            if op(data_lims[i], self.x[i]):
                self.x[i] = data_lims[i]
            else:
                self.x[i] = (1 - ad) * data_lims[i] + ad * self.x[i]

        for op, i in zip(ops, idxs):
            if op(self.x[i], self.y[i]):
                self.y[i] = (1 - ag) * self.x[i] + ag * self.y[i]
            else:
                self.y[i] = self.x[i]

        for op, i, j in zip(ops, idxs, reversed(idxs)):
            y_hyst = self.y[i] + (self.y[i] - self.y[j]) * (1 / (1 - self.hyst) - 1)

            if op(self.y[i], self.z[i]):
                self.z[i] = self.y[i]
            elif op(self.z[i], y_hyst):
                self.z[i] = y_hyst

        return self.z


pg_phase_ticks = [
    list(zip(np.linspace(-np.pi, np.pi, 5), ["-π", "-π/2", "0", "π/2", "π"])),
    [(x, "") for x in np.linspace(-np.pi, np.pi, 9)],
]


def pg_setup_polar_plot(plot, max_r=1):
    plot.showAxis("left", False)
    plot.showAxis("bottom", False)
    plot.setAspectLocked()
    plot.disableAutoRange()
    s = 1.15
    plot.setXRange(-s * max_r, s * max_r)
    plot.setYRange(-s * max_r, s * max_r)

    for i, r in enumerate(np.linspace(0, max_r, 5)[1:]):
        circle = pg.QtGui.QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
        circle.setPen(pg.mkPen("k" if i == 3 else 0.5))
        plot.addItem(circle)

        if i == 3:
            text_item = pg.TextItem(text=str(r), color="k", anchor=(0, 1))
            text_item.setPos(np.cos(np.pi / 8) * r, np.sin(np.pi / 8) * r)
            plot.addItem(text_item)

    for i in range(8):
        deg = (360 / 8) * i
        rad = np.radians(deg)
        x = np.cos(rad)
        y = np.sin(rad)
        text = str(int(deg)) + '\u00b0'
        ax = (-x + 1) / 2
        ay = (y + 1) / 2
        text_item = pg.TextItem(text, color="k", anchor=(ax, ay))
        text_item.setPos(max_r * x * 1.02, max_r * y * 1.02)
        plot.addItem(text_item)
        plot.plot([0, max_r * x], [0, max_r * y], pen=pg.mkPen(0.5))


def pg_mpl_cmap(name):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(name)
    cmap._init()
    return np.array(cmap._lut) * 255


class FreqCounter:
    def __init__(self, a=0.95, num_bits=None):
        self.a = a
        self.num_bits = num_bits
        self.last_t = None
        self.lp_dt = None

    def tick(self):
        now = time.time()

        if self.last_t:
            dt = now - self.last_t
            if self.lp_dt:
                self.lp_dt = self.a * self.lp_dt + (1 - self.a) * dt
                f = 1 / self.lp_dt
                dt_ms = self.lp_dt * 1e3
                if self.num_bits:
                    data_rate = self.num_bits * f * 1e-6
                    s = " {:5.1f} ms, {:5.1f} Hz, {:5.2f} Mbit/s".format(dt_ms, f, data_rate)
                    print(s, end="\r")
                else:
                    print(" {:5.1f} ms, {:5.1f} Hz".format(dt_ms, f), end="\r")
            else:
                self.lp_dt = dt

        self.last_t = now


def get_range_depths(sensor_config: cb.SensorConfig, session_info: dict) -> np.ndarray:
    """Get range depths in meters."""

    range_start = session_info["range_start_m"]
    range_end = range_start + session_info["range_length_m"]

    if sensor_config.mode == Mode.SPARSE:
        num_depths = session_info["data_length"] // sensor_config.sweeps_per_frame
    elif sensor_config.mode == Mode.POWER_BINS:
        num_depths = session_info["bin_count"]
    else:
        num_depths = session_info["data_length"]

    return np.linspace(range_start, range_end, num_depths)


def hex_to_rgb_tuple(hex_color):
    return struct.unpack("BBB", bytes.fromhex(hex_color.lstrip("#")))


def is_power_of_2(n):
    return (n & (n - 1) == 0) and n != 0


def pg_curve_set_data_with_nan(curve, x, y):
    # Workaround for PyQt5 related bug in PyQtGraph where curves don't render if they contain NaNs
    # https://github.com/SainsburyWellcomeCentre/lasagna/issues/247
    # https://github.com/pyqtgraph/pyqtgraph/issues/1057
    # https://github.com/pyqtgraph/pyqtgraph/issues/1057#issuecomment-543190337

    z = np.array(y)
    finite = np.isfinite(z)
    z[~finite] = 0
    connect = np.logical_and(finite, np.roll(finite, -1))
    return curve.setData(x, z, connect=connect)


def optional_or_else(value, default):
    return default if value is None else value
