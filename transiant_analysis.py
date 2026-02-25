import sys
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QHBoxLayout, QMessageBox
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
from scipy.fft import fft, fftfreq, rfft, rfftfreq
import pandas as pd
from datetime import datetime

class DAQGui(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 설정값
        self.DEV = "Dev1"
        self.AI_CH = f"{self.DEV}/ai0"
        self.FS = 1000
        self.BUFFER_SIZE = 50
        self.MIDDLE_REF = 68 
        
        self.full_buffer = np.zeros(self.FS * 12)
        
        self.is_monitoring = False
        self.is_ready_mode = False
        self.start_pos = None      
        
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_data)
        
    def init_ui(self):
        self.setWindowTitle("Baumer DAQ - Zero Set & CSV Save")
        self.setGeometry(100, 100, 1000, 900)
        
        layout = QVBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        # --- 그래프 영역 (기존과 동일) ---
        self.graph_time = pg.PlotWidget(title="Time Domain (Last 1s)")
        self.graph_time.setBackground('w')
        self.graph_time.setYRange(-30, 30)
        self.curve_time = self.graph_time.plot(pen=pg.mkPen('b', width=1))
        layout.addWidget(self.graph_time)
        
        self.graph_fft = pg.PlotWidget(title="Frequency Domain (Real-time)")
        self.graph_fft.setBackground('w')
        self.graph_fft.setXRange(0, 20) 
        self.curve_fft = self.graph_fft.plot(pen=pg.mkPen('r', width=2))
        self.peak_label = pg.TextItem(anchor=(0, 1), color='k', fill=(255, 255, 255, 200))
        self.graph_fft.addItem(self.peak_label)
        layout.addWidget(self.graph_fft)
        
        # --- 버튼 레이아웃 ---
        btn_layout = QHBoxLayout()
        
        self.btn_monitor = QPushButton("연속표시 시작")
        self.btn_monitor.setFixedHeight(50)
        self.btn_monitor.clicked.connect(self.toggle_monitor)
        btn_layout.addWidget(self.btn_monitor)
        
        self.btn_zero = QPushButton("Zero Set")
        self.btn_zero.setFixedHeight(50)
        self.btn_zero.clicked.connect(self.set_zero)
        btn_layout.addWidget(self.btn_zero)
        
        self.btn_ready = QPushButton("READY (Wait 5mm)")
        self.btn_ready.setFixedHeight(50)
        self.btn_ready.clicked.connect(self.activate_ready)
        btn_layout.addWidget(self.btn_ready)

        # ★ CSV 저장 버튼 추가 ★
        self.btn_csv = QPushButton("CSV 데이터 저장")
        self.btn_csv.setFixedHeight(50)
        self.btn_csv.setStyleSheet("background-color: #dcf8c6; font-weight: bold;")
        self.btn_csv.clicked.connect(self.save_to_csv)
        btn_layout.addWidget(self.btn_csv)
        
        layout.addLayout(btn_layout)

    def save_to_csv(self):
        """현재 버퍼에 쌓인 Raw 데이터를 CSV로 저장"""
        try:
            # 파일명 생성 (예: daq_data_20240520_143005.csv)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"daq_raw_data_{timestamp}.csv"
            
            # 시간 축 생성 (현재 12초 버퍼 기준)
            time_axis = np.linspace(0, len(self.full_buffer)/self.FS, len(self.full_buffer))
            
            # DataFrame 생성 및 저장
            df = pd.DataFrame({
                'Time(s)': time_axis,
                'Distance(mm)': self.full_buffer
            })
            
            df.to_csv(filename, index=False)
            QMessageBox.information(self, "저장 완료", f"데이터가 성공적으로 저장되었습니다.\n파일명: {filename}")
            print(f"Data saved to {filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "저장 실패", f"에러 발생: {e}")

    # --- 이하 기존 메서드 (동일) ---
    def set_zero(self):
        if self.is_monitoring:
            current_raw_avg = np.mean(self.full_buffer[-500:]) + self.MIDDLE_REF
            self.MIDDLE_REF = current_raw_avg
            print(f"영점 조절 완료: {self.MIDDLE_REF:.2f}")

    def activate_ready(self):
        if not hasattr(self, 'task'): self.start_task()
        self.is_ready_mode = True
        self.is_monitoring = True
        self.start_pos = None  
        self.btn_ready.setText("기준 위치 측정 중...")
        self.btn_ready.setStyleSheet("background-color: #ffd700;")
        self.timer.start(10)

    def start_task(self):
        try:
            self.task = nidaqmx.Task()
            self.task.ai_channels.add_ai_voltage_chan(self.AI_CH)
            self.task.timing.cfg_samp_clk_timing(rate=self.FS, sample_mode=AcquisitionType.CONTINUOUS)
            self.task.start()
        except Exception as e: print(f"DAQ Start Error: {e}")

    def stop_all(self):
        self.timer.stop()
        self.is_monitoring = False
        self.is_ready_mode = False
        if hasattr(self, 'task'):
            try:
                self.task.stop()
                self.task.close()
            except: pass
            del self.task
        self.btn_monitor.setText("연속표시 시작")
        self.btn_ready.setText("READY (Wait 5mm Move)")
        self.btn_ready.setStyleSheet("")

    def toggle_monitor(self):
        if not self.is_monitoring:
            self.start_task()
            self.is_monitoring = True
            self.btn_monitor.setText("연속표시 중지")
            self.timer.start(20)
        else:
            self.stop_all()

    def process_data(self):
        try:
            raw_v = self.task.read(number_of_samples_per_channel=self.BUFFER_SIZE)
            new_samples = (np.array(raw_v) * 10.4) + 16 - self.MIDDLE_REF
            
            self.full_buffer = np.roll(self.full_buffer, -self.BUFFER_SIZE)
            self.full_buffer[-self.BUFFER_SIZE:] = new_samples
            
            self.curve_time.setData(self.full_buffer[-self.FS:])
            self.update_fft(self.full_buffer[-self.FS*10:])
            
            if self.is_ready_mode:
                if self.start_pos is None:
                    self.start_pos = np.mean(new_samples)
                    return
                displacements = np.abs(new_samples - self.start_pos)
                if np.any(displacements >= 5.0):
                    self.is_ready_mode = False
                    self.handle_trigger()
                    
        except Exception as e:
            print(f"Process Error: {e}")
            self.stop_all()

    def update_fft(self, data):
        n = len(data)
        detrended_data = data - np.mean(data)
        yf = fft(detrended_data)
        xf = fftfreq(n, 1 / self.FS)
        pos_mask = (xf >= 0) & (xf <= 25)
        self.curve_fft.setData(xf[pos_mask], (2.0 / n) * np.abs(yf[pos_mask]))

    def handle_trigger(self):
        self.btn_ready.setText("데이터 수집 중...")
        self.btn_ready.setStyleSheet("background-color: #ff4500; color: white;")
        QTimer.singleShot(4500, self.show_plot)

    def show_plot(self):
        # (기존의 FFT 진폭 추출 그래프 코드와 동일)
        try:
            final_data = self.full_buffer[-(self.FS * 5):]
            self.stop_all()
            t = np.linspace(-0.5, 4.5, len(final_data))
            
            window_size = int(self.FS * 0.2)
            step_size = int(self.FS * 0.05)
            
            fft_times, fft_amplitudes = [], []
            for i in range(0, len(final_data) - window_size, step_size):
                window_data = (final_data[i : i + window_size] - np.mean(final_data[i : i + window_size])) * np.hanning(window_size)
                yf = rfft(window_data)
                fft_amplitudes.append(np.max(np.abs(yf)) * 2 / window_size)
                fft_times.append(t[i + window_size // 2])

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            ax1.plot(t, final_data, label='Raw Distance')
            ax1.set_title("Triggered Data")
            ax2.plot(fft_times, fft_amplitudes, color='g', label='FFT Amplitude')
            ax2.set_title("Time-Amplitude (FFT)")
            plt.tight_layout()
            plt.show()
        except Exception as e: print(f"Plot Error: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DAQGui()
    win.show()
    sys.exit(app.exec_())