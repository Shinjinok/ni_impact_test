import nidaqmx
from nidaqmx.constants import AcquisitionType, ExcitationSource, TerminalConfiguration
import numpy as np
import time
from datetime import datetime
from collections import deque
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# try importing a Qt binding; the user may need to install one
try:
    from PyQt5 import QtWidgets
    from PyQt5 import QtGui
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    QtWidgets = None
    QtGui = None
    FigureCanvas = None
    # we'll check later and print a helpful message
# --- 설정 (전달해주신 센서 감도 반영) ---
physical_channel = "cDAQ1Mod4/ai0"
physical_channel_hammer = "cDAQ1Mod4/ai1"
sample_rate = 25600
chunk_size = 1024
threshold = 10.0  # 단위가 Newton으로 바뀌었으므로 임계값 상향 조정 필요
simulation = False  # 시뮬레이션 모드 여부

# 키보드 트리거 플래그
trigger_flag = False
# 종료 요청 플래그 (버튼/창 닫기)
stop_flag = False

# 마지막 캡처 데이터 저장용 전역 변수
last_impact_data = None  # (cap_ai0, cap_ai1, t)
last_fft_data = None     # (xf_0, mag_0)

def calculate_fft(signal, sample_rate):
    N = len(signal)
    #windowed_signal = signal * np.hanning(N)
    #yf = fft(windowed_signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / sample_rate)[:N//2]
    target_yf = np.asarray(yf[0:N//2])
    magnitude = 2.0 / N * np.abs(target_yf)
    return xf, magnitude

def save_data():
    """Save last captured impact and FFT data to CSV files."""
    global last_impact_data, last_fft_data
    
    if last_impact_data is None or last_fft_data is None:
        print("No data to save. Please trigger a capture first.")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save impact data
    cap_ai0, cap_ai1, t = last_impact_data
    impact_filename = f"impact_{timestamp}.csv"
    try:
        impact_data = np.column_stack((t, cap_ai0, cap_ai1))
        np.savetxt(impact_filename, impact_data, delimiter=',', 
                   header='Time(s),Sensor(g),Hammer(N)', comments='')
        print(f"Impact data saved to {impact_filename}")
    except Exception as e:
        print(f"Error saving impact data: {e}")
    
    # Save FFT data
    xf_0, mag_0 = last_fft_data
    fft_filename = f"fft_{timestamp}.csv"
    try:
        fft_data = np.column_stack((xf_0, mag_0))
        np.savetxt(fft_filename, fft_data, delimiter=',', 
                   header='Frequency(Hz),Magnitude', comments='')
        print(f"FFT data saved to {fft_filename}")
    except Exception as e:
        print(f"Error saving FFT data: {e}")

def plot_export():
    """Export last captured data to a new matplotlib window."""
    global last_impact_data, last_fft_data
    
    if last_impact_data is None or last_fft_data is None:
        print("No data to plot. Please trigger a capture first.")
        return
    
    cap_ai0, cap_ai1, t = last_impact_data
    xf_0, mag_0 = last_fft_data
    
    # Create a new independent figure (separate from Qt dialog)
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(hspace=0.4)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    
    # Plot impact data
    ax1_right = ax1.twinx()
    line1 = ax1.plot(t, cap_ai0, label='Sensor (g)', color='blue')
    ax1.set_ylabel('Sensor (g)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    line2 = ax1_right.plot(t, cap_ai1, label='Hammer (N)', color='orange', alpha=0.7)
    ax1_right.set_ylabel('Hammer (N)', color='orange')
    ax1_right.yaxis.set_label_position("right")
    ax1_right.tick_params(axis='y', labelcolor='orange')
    
    ax1.set_title(f"Impact Event Data (Duration: {t[-1] - t[0]:.2f}s)")
    ax1.set_xlabel('Time (s)')
    ax1.grid(True)
    
    # Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    # Plot FFT data
    ax2.plot(xf_0, mag_0, color='blue', label='Magnitude', alpha=0.7)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title("FFT Analysis")
    ax2.grid(True)
    ax2.legend()
    
    # Show the new figure (non-blocking for Qt backend)
    fig.canvas.manager.show()


class PlotDialog(QtWidgets.QDialog if QtWidgets else object):
    """Qt dialog containing a matplotlib figure with two subplots and a trigger button."""

    def __init__(self, parent=None):
        if QtWidgets is None:
            # fallback to no-op if Qt isn't installed
            raise RuntimeError("Qt binding not available")
        super().__init__(parent)
        self.setWindowTitle("Impact Data")

        # create the figure and axes just like before
        self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plt.subplots_adjust(hspace=0.4)
        self.ax1_right = self.ax1.twinx()

        # embed into canvas
        self.canvas = FigureCanvas(self.figure)

        # trigger button
        self.trigger_button = QtWidgets.QPushButton("Trigger")
        self.trigger_button.clicked.connect(self._on_trigger_clicked)

        # exit button
        self.exit_button = QtWidgets.QPushButton("Exit")
        self.exit_button.clicked.connect(self._on_exit_clicked)

        # save button
        self.save_button = QtWidgets.QPushButton("Save Data")
        self.save_button.clicked.connect(self._on_save_clicked)

        # plot export button
        self.plot_button = QtWidgets.QPushButton("Plot Export")
        self.plot_button.clicked.connect(self._on_plot_clicked)

        # capture time input
        time_layout = QtWidgets.QHBoxLayout()
        self.capture_edit = QtWidgets.QLineEdit("1.0")
        self.capture_edit.setMaximumWidth(60)
        self.capture_edit.setToolTip("Enter capture duration in seconds (0.1~10.0)")
        # enforce numeric input with range (validator lives in QtGui)
        if QtGui is not None:
            validator = QtGui.QDoubleValidator(0.1, 10.0, 2, self)
            validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
            self.capture_edit.setValidator(validator)
        time_label = QtWidgets.QLabel("Capture time (s):")
        range_label = QtWidgets.QLabel("0.1 ~ 10.0")
        time_layout.addWidget(time_label)
        time_layout.addWidget(self.capture_edit)
        time_layout.addWidget(range_label)
        time_layout.addStretch()

        # max frequency input
        freq_layout = QtWidgets.QHBoxLayout()
        self.max_freq_edit = QtWidgets.QLineEdit("500")
        self.max_freq_edit.setMaximumWidth(60)
        self.max_freq_edit.setToolTip("Enter max frequency for FFT display (100~5000)")
        # enforce numeric input with range
        if QtGui is not None:
            freq_validator = QtGui.QDoubleValidator(100.0, 5000.0, 1, self)
            freq_validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
            self.max_freq_edit.setValidator(freq_validator)
        freq_label = QtWidgets.QLabel("Max Frequency (Hz):")
        freq_range_label = QtWidgets.QLabel("100 ~ 5000")
        freq_layout.addWidget(freq_label)
        freq_layout.addWidget(self.max_freq_edit)
        freq_layout.addWidget(freq_range_label)
        freq_layout.addStretch()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addLayout(time_layout)
        layout.addLayout(freq_layout)

        # place buttons on a horizontal row
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.trigger_button)
        btn_layout.addWidget(self.save_button)
        btn_layout.addWidget(self.plot_button)
        btn_layout.addWidget(self.exit_button)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def _on_trigger_clicked(self):
        global trigger_flag
        trigger_flag = True

    def _on_exit_clicked(self):
        global stop_flag
        stop_flag = True
        self.close()

    def _on_save_clicked(self):
        save_data()

    def _on_plot_clicked(self):
        plot_export()

    def draw(self):
        """Convenience wrapper for drawing and processing Qt events."""
        self.canvas.draw()
        QtWidgets.QApplication.processEvents()

def continuous_acquisition():
    global trigger_flag, simulation

    # if Qt support available, create a dialog for plotting
    dialog = None
    if QtWidgets is not None:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        dialog = PlotDialog()
        dialog.show()
    else:
        print("WARNING: Qt binding not installed. The graph will use the default matplotlib window.")

    with nidaqmx.Task() as task:
        try:
            # 가속도계 (5mV/g -> 5.0)
            task.ai_channels.add_ai_accel_chan(physical_channel, units=nidaqmx.constants.AccelUnits.G,
                                            sensitivity=5.0, current_excit_source=ExcitationSource.INTERNAL,
                                            current_excit_val=0.004)
            # 해머 (2.2mV/N -> 2.2)
            task.ai_channels.add_ai_force_iepe_chan(physical_channel_hammer, units=nidaqmx.constants.ForceUnits.NEWTONS,
                                                sensitivity=2.2, current_excit_source=ExcitationSource.INTERNAL,
                                                current_excit_val=0.004)

            task.timing.cfg_samp_clk_timing(rate=sample_rate, samps_per_chan=chunk_size,
                                            sample_mode=AcquisitionType.CONTINUOUS)
        except Exception as e:
            simulation = True
            print("Hardware configuration failed. Switching to simulation mode.")
            

        # determine capture duration from dialog (seconds)
        capture_time = 1.0
        if dialog is not None:
            try:
                capture_time = float(dialog.capture_edit.text())
            except Exception:
                pass
        # clamp to 0.1 - 10.0 seconds
        capture_time = max(0.1, min(10.0, capture_time))
        prev_capture_time = capture_time  # track previous value for change detection
        pre_samples = int(sample_rate * 10 / 1000.0)  # always keep 10ms pre-trigger buffer
        total_samples = int(sample_rate * capture_time)
        post_samples = total_samples - pre_samples
        pre_buffer_ai0 = deque(maxlen=pre_samples)
        pre_buffer_ai1 = deque(maxlen=pre_samples)

        # 그래프 초기 설정
        if dialog is not None:
            fig = dialog.figure
            ax1 = dialog.ax1
            ax2 = dialog.ax2
            ax1_right = dialog.ax1_right
        else:
            plt.ion()
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            plt.subplots_adjust(hspace=0.4)
            ax1_right = ax1.twinx()
            mngr = plt.get_current_fig_manager()
            mngr.window.geometry("+0+0")

        print(f">>> Ready for first impact! Click 'Trigger' button to capture (capture {capture_time:.2f}s)...")

        try:
            while True:
                # check for exit request
                if stop_flag:
                    print("Exit requested, stopping acquisition.")
                    break
                # also if dialog was closed
                if dialog is not None and not dialog.isVisible():
                    print("Dialog closed, stopping acquisition.")
                    break
                
                # check if capture_time has changed and reinitialize if needed
                new_capture_time = capture_time
                if dialog is not None:
                    try:
                        new_capture_time = float(dialog.capture_edit.text())
                    except Exception:
                        pass
                new_capture_time = max(0.1, min(10.0, new_capture_time))
                
                if new_capture_time != prev_capture_time:
                    print(f"Capture time changed from {prev_capture_time:.2f}s to {new_capture_time:.2f}s")
                    capture_time = new_capture_time
                    prev_capture_time = capture_time
                    # reinitialize buffers and parameters
                    total_samples = int(sample_rate * capture_time)
                    post_samples = total_samples - pre_samples
                    pre_buffer_ai0.clear()
                    pre_buffer_ai1.clear()
                    # reinitialize graph axes
                    ax1.cla()
                    ax1_right.cla()
                    ax2.cla()
                    if dialog is not None:
                        dialog.draw()
                    else:
                        fig.canvas.draw()
                
                # matplotlib 이벤트 처리
                fig.canvas.flush_events()
                
                # 데이터 수집 모드
                if not simulation:
                    data = task.read(number_of_samples_per_channel=chunk_size)
                else:
                    # 시뮬레이션 모드에서는 10Hz, 15Hz, 100Hz 정현파를 혼합한 신호 생성
                    t_chunk = np.arange(chunk_size) / sample_rate
                    # 센서 채널 (g 단위): 작은 진폭의 혼합 사인파 + 약한 랜덤 노이즈
                    sensor_signal = (
                        1 * np.sin(2 * np.pi * 10 * t_chunk)
                        + 1 * np.sin(2 * np.pi * 200 * t_chunk)
                        + 1 * np.sin(2 * np.pi * 300 * t_chunk)
                    )
                    sensor_signal += 0.02 * np.random.normal(0, 1, chunk_size)

                    # 해머 채널 (N 단위): 동일 주파수 성분을 포함하되 진폭을 키워 임계값을 넘기기 쉽게 함
                    hammer_signal = 0 * (
                        0.6 * np.sin(2 * np.pi * 10 * t_chunk)
                        + 0.3 * np.sin(2 * np.pi * 15 * t_chunk)
                        + 0.1 * np.sin(2 * np.pi * 100 * t_chunk)
                    )
                    hammer_signal += 0.5 * np.random.normal(0, 1, chunk_size)

                    data = [sensor_signal, hammer_signal]
                    # time.sleep(chunk_size / sample_rate)

                data_ai0 = np.array(data[0])
                data_ai1 = np.array(data[1])

                for v0, v1 in zip(data_ai0, data_ai1):
                    pre_buffer_ai0.append(v0)
                    pre_buffer_ai1.append(v1)

                # 트리거 감지
                if not simulation:
                    crossings = np.where(data_ai0 >= threshold)[0]
                else:
                    # Simulation mode: trigger only via button click
                    if trigger_flag:
                        crossings = np.array([chunk_size // 2])
                    else:
                        crossings = np.array([])

                if crossings.size > 0 or trigger_flag:
                    trigger_flag = False
                    idx = crossings[0]
                    print(f"Trigger detected! Capturing {capture_time:.2f}s...")
                    if not simulation:
                        remaining = max(0, post_samples - (chunk_size - idx))
                        raw_post = task.read(number_of_samples_per_channel=remaining)
                        
                        cap_ai0 = np.concatenate([list(pre_buffer_ai0), data_ai0[idx:], raw_post[0]])
                        cap_ai1 = np.concatenate([list(pre_buffer_ai1), data_ai1[idx:], raw_post[1]])
                    else:
                        # For simulation mode, synthesize a full 1.0s capture (pre + post)
                        total_len = pre_samples + post_samples

                        # tail from current chunk starting at trigger index
                        tail = data_ai0[idx:]
                        tail_h = data_ai1[idx:]

                        # how many additional samples we need after taking pre_buffer and tail
                        current_len = len(pre_buffer_ai0) + len(tail)
                        need = max(0, total_len - current_len)

                        # generate the remaining post samples
                        if need > 0:
                            t_post = np.arange(need) / sample_rate
                            post_sensor = (
                                1 * np.sin(2 * np.pi * 10 * t_post)
                                + 1 * np.sin(2 * np.pi * 200 * t_post)
                                + 1 * np.sin(2 * np.pi * 300 * t_post)
                            )
                            post_sensor = post_sensor + 0.02 * np.random.normal(0, 1, need)

                            post_hammer = 0.5 * np.random.normal(0, 1, need)
                        else:
                            post_sensor = np.array([])
                            post_hammer = np.array([])

                        cap_ai0 = np.concatenate([list(pre_buffer_ai0), tail, post_sensor])
                        cap_ai1 = np.concatenate([list(pre_buffer_ai1), tail_h, post_hammer])

                        # ensure exact length (trim or pad with small noise)
                        expected = total_len
                        if len(cap_ai0) > expected:
                            cap_ai0 = cap_ai0[:expected]
                            cap_ai1 = cap_ai1[:expected]
                        elif len(cap_ai0) < expected:
                            pad = expected - len(cap_ai0)
                            cap_ai0 = np.concatenate([cap_ai0, 0.02 * np.random.normal(0, 1, pad)])
                            cap_ai1 = np.concatenate([cap_ai1, 0.5 * np.random.normal(0, 1, pad)])
                    
                    # compute total length and time vector based on capture_time
                    total_len = len(cap_ai0)
                    # ignore pre-trigger offset; start at 0 and go to capture_time
                    t = np.linspace(0.0, capture_time, total_len)

                    # 그래프 업데이트
                    ax1.cla()
                    # 센서값은 왼쪽 y축, 해머값은 오른쪽 y축
                    line1 = ax1.plot(t, cap_ai0, label='Sensor (g)', color='blue')
                    ax1.set_ylabel('Sensor (g)', color='blue')
                    ax1.tick_params(axis='y', labelcolor='blue')
                    
                    ax1_right.cla()  # 이전 데이터 삭제
                    line2 = ax1_right.plot(t, cap_ai1, label='Hammer (N)', color='orange', alpha=0.7)
                    # 오른쪽 레이블 설정: set_label_position('right')은 기본값이지만, 
                    # 명확하게 하기 위해 텍스트 정렬 및 위치를 확인합니다.
                    ax1_right.set_ylabel('Hammer (N)', color='orange')
                    ax1_right.yaxis.set_label_position("right") # 레이블을 오른쪽에 고정
                    ax1_right.tick_params(axis='y', labelcolor='orange')
                    
                    # title includes capture time
                    ax1.set_title(f"Impact Event Captured ({capture_time:.2f}s)")
                    ax1.set_xlabel('Time (s)')
                    
                    # set x-axis limits based purely on capture_time
                    ax1.set_xlim(0, capture_time)
                    
                    # 범례 표시
                    lines = line1 + line2
                    labels = [l.get_label() for l in lines]
                    ax1.legend(lines, labels, loc='upper right')
                    
                    # --- 2. FFT graph and peak display ---
                    xf_0, mag_0 = calculate_fft(cap_ai0, sample_rate)
                    ax2.cla()
                    ax2.plot(xf_0, mag_0, color='blue', label='Magnitude', alpha=0.7)

                    # read max frequency from dialog
                    if dialog is not None:
                        try:
                            max_freq = float(dialog.max_freq_edit.text())
                            max_freq = max(100.0, min(5000.0, max_freq))
                        except Exception:
                            max_freq = 500.0
                    else:
                        max_freq = 500.0

                    # Filter data within 0~max_freq range
                    mask = (xf_0 >= 0) & (xf_0 <= max_freq)
                    f_range = xf_0[mask]
                    m_range = mag_0[mask]

                    # Find peaks (scipy.signal.find_peaks)
                    peaks, properties = find_peaks(m_range, height=0)
                    peak_heights = properties['peak_heights']

                    if len(peaks) > 0:
                        # Sort by amplitude, extract top 4
                        top_indices = np.argsort(peak_heights)[-4:][::-1]
                        
                        # Color map for distinguishing peaks
                        colors = ['red', 'green', 'purple', 'brown']

                        for i, idx in enumerate(top_indices):
                            p_idx = peaks[idx]
                            freq = f_range[p_idx]
                            val = m_range[p_idx]
                            
                            # Peak point circle
                            ax2.scatter(freq, val, color=colors[i % 4], s=40, zorder=5)
                            
                            # Frequency text display (staggered to avoid overlap)
                            ax2.annotate(f'{freq:.1f}Hz', 
                                        xy=(freq, val), 
                                        xytext=(freq + 5, val + (max(m_range)*0.02)),
                                        fontsize=9, 
                                        color=colors[i % 4],
                                        fontweight='bold')

                    ax2.set_xlim(0, max_freq)
                    ax2.set_xlabel('Frequency (Hz)')
                    ax2.set_ylabel('Magnitude')
                    ax2.set_title(f"Top 4 FFT Peaks (Max: {max_freq:.0f}Hz)")
                    ax2.grid(True)
                    
                    if dialog is not None:
                        dialog.draw()
                    else:
                        fig.canvas.draw()
                        plt.pause(0.01)
                    
                    # Save data to global variables for save button
                    global last_impact_data, last_fft_data
                    last_impact_data = (cap_ai0, cap_ai1, t)
                    last_fft_data = (xf_0, mag_0)
                    
                    print(">>> Measurement Finished.")
                    
                    pre_buffer_ai0.clear()
                    pre_buffer_ai1.clear()
                
                time.sleep(0.01)
                # process Qt events so window remains responsive
                if dialog is not None:
                    QtWidgets.QApplication.processEvents()
        
        except KeyboardInterrupt:
            print("Stopped.")

if __name__ == "__main__":
    continuous_acquisition()