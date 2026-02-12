import nidaqmx
from nidaqmx.constants import AcquisitionType, ExcitationSource, TerminalConfiguration
import numpy as np
import time
from collections import deque
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
# --- 설정 (전달해주신 센서 감도 반영) ---
physical_channel = "cDAQ1Mod4/ai0"
physical_channel_hammer = "cDAQ1Mod4/ai1"
sample_rate = 25600
chunk_size = 1024
threshold = 10.0  # 단위가 Newton으로 바뀌었으므로 임계값 상향 조정 필요
simulation = False  # 시뮬레이션 모드 여부

# 키보드 트리거 플래그
trigger_flag = False

def on_key_press(event):
    """키보드 이벤트 핸들러 - 'p' 키 감지"""
    global trigger_flag
    if event.key == 'p':
        trigger_flag = True

def calculate_fft(signal, sample_rate):
    N = len(signal)
    windowed_signal = signal * np.hanning(N)
    yf = fft(windowed_signal)
    xf = fftfreq(N, 1 / sample_rate)[:N//2]
    target_yf = np.asarray(yf[0:N//2])
    magnitude = 2.0 / N * np.abs(target_yf)
    return xf, magnitude

def continuous_acquisition():
    global trigger_flag
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
           # simulation = True
            print("Hardware configuration failed. Switching to simulation mode.")
            

        pre_samples = int(sample_rate * 10 / 1000.0)
        post_samples = int(sample_rate * 990 / 1000.0)
        pre_buffer_ai0 = deque(maxlen=pre_samples)
        pre_buffer_ai1 = deque(maxlen=pre_samples)

        # 그래프 초기 설정
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plt.subplots_adjust(hspace=0.4)
        
        # ax1_right 미리 생성 (중복 생성 방지)
        ax1_right = ax1.twinx()
        
        # 창을 좌측 상단에 배치
        mngr = plt.get_current_fig_manager()
        mngr.window.geometry("+0+0")
        # 키보드 이벤트 핸들러 연결
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        if not simulation:
            print(">>> Ready for first impact! Waiting for trigger...")
        else:
            print(">>> Ready for first impact! Press 'p' to trigger capture...")

        try:
            while True:
                # matplotlib 이벤트 처리
                fig.canvas.flush_events()
                
                # 데이터 수집 모드
                if not simulation:
                    data = task.read(number_of_samples_per_channel=chunk_size)
                else:
                    # 시뮬레이션 모드에서는 임의의 데이터 생성
                    data = [np.random.normal(0, 1, chunk_size), np.random.normal(0, 1, chunk_size)]
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
                    # 시뮬레이션 모드에서는 임의의 트리거 발생
                    if trigger_flag:
                        crossings = np.array([chunk_size // 2])
                    else:
                        crossings = np.where(data_ai1 >= threshold)[0]

                if crossings.size > 0 or trigger_flag:
                    trigger_flag = False
                    idx = crossings[0]
                    print(f"Trigger detected! Capturing 1.0s...")
                    if not simulation:
                        remaining = max(0, post_samples - (chunk_size - idx))
                        raw_post = task.read(number_of_samples_per_channel=remaining)
                        
                        cap_ai0 = np.concatenate([list(pre_buffer_ai0), data_ai0[idx:], raw_post[0]])
                        cap_ai1 = np.concatenate([list(pre_buffer_ai1), data_ai1[idx:], raw_post[1]])
                    else:
                       # remaining = max(0, post_samples - (chunk_size - idx))
                       # raw_post = task.read(number_of_samples_per_channel=remaining)
                        
                        cap_ai0 = np.concatenate([list(pre_buffer_ai0), data_ai0[idx:]])#, raw_post[0]])
                        cap_ai1 = np.concatenate([list(pre_buffer_ai1), data_ai1[idx:]])#, raw_post[1]])
                    
                    t = (np.arange(len(cap_ai0)) - len(pre_buffer_ai0)) / sample_rate

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
                    
                    ax1.set_title("Impact Event Captured")
                    ax1.set_xlabel('Time (s)')
                    
                    # 범례 표시
                    lines = line1 + line2
                    labels = [l.get_label() for l in lines]
                    ax1.legend(lines, labels, loc='upper right')
                    
                    ax1.grid(True)
                    
                    # --- 2. FFT 그래프 및 피크 표시 ---
                    xf_0, mag_0 = calculate_fft(cap_ai0, sample_rate)
                    ax2.cla()
                    ax2.plot(xf_0, mag_0, color='blue', label='Magnitude', alpha=0.7)

                    # 1. 0~500Hz 범위 내의 데이터만 필터링
                    mask = (xf_0 >= 0) & (xf_0 <= 500)
                    f_range = xf_0[mask]
                    m_range = mag_0[mask]

                    # 2. 피크 찾기 (scipy.signal.find_peaks 사용)
                    # height=0으로 설정하면 모든 양수 피크를 찾습니다.
                    peaks, properties = find_peaks(m_range, height=0)
                    peak_heights = properties['peak_heights']

                    if len(peaks) > 0:
                        # 3. 진폭(height)이 큰 순서대로 인덱스 정렬 후 상위 4개 추출
                        top_indices = np.argsort(peak_heights)[-4:][::-1]
                        
                        # 컬러 맵 (여러 피크를 구분하기 위함, 선택 사항)
                        colors = ['red', 'green', 'purple', 'brown']

                        for i, idx in enumerate(top_indices):
                            p_idx = peaks[idx]
                            freq = f_range[p_idx]
                            val = m_range[p_idx]
                            
                            # 피크 지점 동그라미
                            ax2.scatter(freq, val, color=colors[i % 4], s=40, zorder=5)
                            
                            # 주파수 값 텍스트 표시 (살짝 엇갈리게 배치하여 겹침 방지)
                            ax2.annotate(f'{freq:.1f}Hz', 
                                        xy=(freq, val), 
                                        xytext=(freq + 5, val + (max(m_range)*0.02)), # 약간 위쪽으로
                                        fontsize=9, 
                                        color=colors[i % 4],
                                        fontweight='bold')

                    ax2.set_xlim(0, 500)
                    ax2.set_xlabel('Frequency (Hz)')
                    ax2.set_ylabel('Magnitude')
                    ax2.set_title("Top 4 FFT Peaks")
                    ax2.grid(True)
                    
                    fig.canvas.draw()
                    plt.pause(0.01)
                    
                    print(">>> Measurement Finished.")
                    
                    pre_buffer_ai0.clear()
                    pre_buffer_ai1.clear()
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("Stopped.")

if __name__ == "__main__":
    continuous_acquisition()