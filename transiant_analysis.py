import sys
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QHBoxLayout
from PyQt5.QtCore import QTimer
import pyqtgraph as pg

class DAQGui(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 설정값
        self.DEV = "Dev1"
        self.AI_CH = f"{self.DEV}/ai0"
        self.FS = 1000
        self.BUFFER_SIZE = 50
        self.MIDDLE_REF = 68
        
        # 5초 데이터를 표시하기 위해 버퍼를 넉넉히 7초 확보
        self.full_buffer = np.zeros(self.FS * 7)
        
        self.is_monitoring = False
        self.is_ready_mode = False
        self.start_pos = None      
        
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_data)
        
    def init_ui(self):
        self.setWindowTitle("Baumer DAQ - 0.5s Pre-Trigger & 5s View")
        self.setGeometry(100, 100, 900, 600)
        
        layout = QVBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setBackground('w')
        self.graph_widget.setYRange(-60, 60)
        self.curve = self.graph_widget.plot(pen=pg.mkPen('b', width=1))
        layout.addWidget(self.graph_widget)
        
        btn_layout = QHBoxLayout()
        self.btn_monitor = QPushButton("연속표시 시작")
        self.btn_monitor.setFixedHeight(50)
        self.btn_monitor.clicked.connect(self.toggle_monitor)
        btn_layout.addWidget(self.btn_monitor)
        
        self.btn_ready = QPushButton("READY (Wait 5mm Move)")
        self.btn_ready.setFixedHeight(50)
        self.btn_ready.clicked.connect(self.activate_ready)
        btn_layout.addWidget(self.btn_ready)
        layout.addLayout(btn_layout)

    def activate_ready(self):
        if not hasattr(self, 'task'): self.start_task()
        self.is_ready_mode = True
        self.is_monitoring = True
        self.start_pos = None  
        self.btn_ready.setText("기준 위치 측정 중...")
        self.btn_ready.setStyleSheet("background-color: #ffd700;")
        self.timer.start(10)

    def start_task(self):
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan(self.AI_CH)
        self.task.timing.cfg_samp_clk_timing(rate=self.FS, sample_mode=AcquisitionType.CONTINUOUS)
        self.task.start()

    def stop_all(self):
        self.timer.stop()
        self.is_monitoring = False
        self.is_ready_mode = False
        if hasattr(self, 'task'):
            self.task.stop()
            self.task.close()
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
            self.curve.setData(self.full_buffer[-(self.FS*1):])
            
            if self.is_ready_mode:
                if self.start_pos is None:
                    self.start_pos = np.mean(new_samples)
                    return

                displacements = np.abs(new_samples - self.start_pos)
                
                if np.any(displacements >= 5.0):
                    print("트리거 감지! 과거 0.5초 포함 5초 수집 시작.")
                    self.is_ready_mode = False
                    self.handle_trigger()
                    
        except Exception as e:
            print(f"Error: {e}")
            self.stop_all()

    def handle_trigger(self):
        # 트리거 시점에 이미 버퍼에 과거 데이터가 있으므로, 
        # 남은 4.5초를 더 수집한 뒤 그래프를 출력합니다.
        remaining_time_ms = 4500 
        self.btn_ready.setText("데이터 수집 중 (5.0s)...")
        self.btn_ready.setStyleSheet("background-color: #ff4500; color: white;")
        
        QTimer.singleShot(remaining_time_ms, self.show_plot)

    def show_plot(self):
        try:
            # 5초 분량의 데이터 추출 (5000 샘플)
            # 현재 시점이 수집 종료 시점이므로 마지막 5000개를 가져옵니다.
            final_data = self.full_buffer[-(self.FS * 5):]
            self.stop_all()
            
            plt.figure(figsize=(12, 6))
            # 트리거 시점은 시작으로부터 0.5초 지점입니다.
            t = np.linspace(-0.5, 4.5, len(final_data)) 
            plt.plot(t, final_data, 'b-', label='Captured Data (Pre 0.5s ~ Post 4.5s)')
            
            # 트리거 발생 시점(0초) 표시
            plt.axvline(0, color='r', linestyle='--', alpha=0.7, label='Trigger Point')
            
            # Y축 자동 스케일 및 여유 공간 설정
            d_min, d_max = np.min(final_data), np.max(final_data)
            pad = (d_max - d_min) * 0.1 if d_max != d_min else 1.0
            plt.ylim(d_min - pad, d_max + pad)
            
            plt.title("Displacement Trigger (0.5s Pre-recorded, 5s Total)")
            plt.xlabel("Time (s) [0 = Trigger Time]")
            plt.ylabel("Distance (mm)")
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.show()
            
        except Exception as e:
            print(f"Plot Error: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DAQGui()
    win.show()
    sys.exit(app.exec_())