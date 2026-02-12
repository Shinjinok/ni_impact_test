import nidaqmx
from nidaqmx.constants import AcquisitionType, ExcitationSource, TerminalConfiguration
import numpy as np
import time
from collections import deque
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# --- 설정 (전달해주신 센서 감도 반영) ---
physical_channel = "cDAQ1Mod4/ai0"
physical_channel_hammer = "cDAQ1Mod4/ai1"
sample_rate = 25600
chunk_size = 1024
threshold = 10.0  # 단위가 Newton으로 바뀌었으므로 임계값 상향 조정 필요

def calculate_fft(signal, sample_rate):
    N = len(signal)
    windowed_signal = signal * np.hanning(N)
    yf = fft(windowed_signal)
    xf = fftfreq(N, 1 / sample_rate)[:N//2]
    target_yf = np.asarray(yf[0:N//2])
    magnitude = 2.0 / N * np.abs(target_yf)
    return xf, magnitude

def continuous_acquisition():
    
    with nidaqmx.Task() as task:
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

        pre_samples = int(sample_rate * 10 / 1000.0)
        post_samples = int(sample_rate * 990 / 1000.0)
        pre_buffer_ai0 = deque(maxlen=pre_samples)
        pre_buffer_ai1 = deque(maxlen=pre_samples)

        # 그래프 초기 설정
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plt.subplots_adjust(hspace=0.4)
        
        # 창을 좌측 상단에 배치
        mngr = plt.get_current_fig_manager()
        mngr.window.geometry("+0+0")

        print(">>> Ready for first impact! Waiting for trigger...")

        try:
            while True:
                # 데이터 수집 모드
                data = task.read(number_of_samples_per_channel=chunk_size)
                data_ai0 = np.array(data[0])
                data_ai1 = np.array(data[1])

                for v0, v1 in zip(data_ai0, data_ai1):
                    pre_buffer_ai0.append(v0)
                    pre_buffer_ai1.append(v1)

                # 트리거 감지
                crossings = np.where(data_ai1 >= threshold)[0]
                if crossings.size > 0:
                    idx = crossings[0]
                    print(f"Trigger detected! Capturing 1.0s...")

                    remaining = max(0, post_samples - (chunk_size - idx))
                    raw_post = task.read(number_of_samples_per_channel=remaining)
                    
                    cap_ai0 = np.concatenate([list(pre_buffer_ai0), data_ai0[idx:], raw_post[0]])
                    cap_ai1 = np.concatenate([list(pre_buffer_ai1), data_ai1[idx:], raw_post[1]])
                    
                    t = (np.arange(len(cap_ai0)) - len(pre_buffer_ai0)) / sample_rate

                    # 그래프 업데이트
                    ax1.cla()
                    ax1.plot(t, cap_ai0, label='Sensor (g)', color='blue')
                    ax1.plot(t, cap_ai1, label='Hammer (N)', color='orange', alpha=0.7)
                    ax1.set_title("Impact Event Captured")
                    ax1.legend()
                    ax1.grid(True)
                    
                    xf_0, mag_0 = calculate_fft(cap_ai0, sample_rate)
                    ax2.cla()
                    ax2.plot(xf_0, mag_0, color='blue', label='Magnitude')
                    ax2.set_xlim(0, 500)
                    ax2.set_xlabel('Frequency (Hz)')
                    ax2.set_ylabel('Magnitude')
                    ax2.set_title("FFT Spectrum")
                    ax2.legend()
                    ax2.grid(True)
                    
                    fig.canvas.draw()
                    plt.pause(0.01)
                    
                    print(">>> Measurement Finished.")
                    
                    pre_buffer_ai0.clear()
                    pre_buffer_ai1.clear()
                
        except KeyboardInterrupt:
            print("Stopped.")

if __name__ == "__main__":
    continuous_acquisition()