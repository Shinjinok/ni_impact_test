import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
from scipy.fft import rfft, rfftfreq

def plot_with_enhanced_labels():
    # 1. 파일 선택
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="CSV 선택", filetypes=[("CSV files", "*.csv")])
    if not file_path: return

    try:
        # 2. 데이터 로드 및 FFT 진폭 계산 (기존 로직 유지)
        df = pd.read_csv(file_path)
        t, dist = df['Time(s)'].values, df['Distance(mm)'].values
        fs = 1 / (t[1] - t[0])

        window_size = int(fs * 0.2)
        step_size = int(fs * 0.05)
        acf = 2.0 
        
        fft_times, fft_amplitudes = [], []
        for i in range(0, len(dist) - window_size, step_size):
            window_data = (dist[i : i + window_size] - np.mean(dist[i : i + window_size])) * np.hanning(window_size)
            yf = rfft(window_data)
            amp = (np.max(np.abs(yf)) * 2 / window_size) * acf
            fft_amplitudes.append(amp)
            fft_times.append(t[i + window_size // 2])

        fft_times = np.array(fft_times)
        fft_amplitudes = np.array(fft_amplitudes)
        threshold = 1.0 

        # 3. 교차점 정밀 계산
        # 부호가 바뀌는 지점 찾기
        cross_indices = np.where(np.diff(np.sign(fft_amplitudes - threshold)))[0]
        
        # 4. 시각화
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        def set_refined_grid(ax):
            ax.grid(True, which='major', linestyle='-', linewidth='0.8', color='gray', alpha=0.7)
            ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray', alpha=0.5)
            ax.minorticks_on()
            ax.tick_params(axis='both', which='both', direction='in')

        for ax in [ax1, ax2]:
            if ax == ax1:
                ax.plot(t, dist, color='dodgerblue', alpha=0.3, label='Original Waveform')
                ax.plot(fft_times, fft_amplitudes, color='crimson', linewidth=2, label='FFT Amplitude')
            else:
                ax.plot(fft_times, fft_amplitudes, color='forestgreen', linewidth=2, label='FFT Peak Amplitude')
                ax.fill_between(fft_times, fft_amplitudes, color='green', alpha=0.1)

            # 1mm 기준선
            ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label='Threshold 1.0mm')
            
            # 교차점 표시 및 라벨링 (화살표 추가)
            for i, idx in enumerate(cross_indices):
                ct = fft_times[idx]
                
                # 교차점 강조 (노란색 점)
                ax.plot(ct, threshold, 'o', markerfacecolor='yellow', markeredgecolor='black', markersize=6, zorder=5)
                
                # 텍스트가 겹치지 않게 위아래 번갈아 배치
                y_offset = 0.5 if i % 2 == 0 else -0.8
                
                ax.annotate(f'{ct:.2f}s', 
                            xy=(ct, threshold), 
                            xytext=(ct, threshold + y_offset),
                            arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.6),
                            fontsize=9, fontweight='bold', color='red',
                            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', alpha=0.8),
                            ha='center')

            ax.set_ylabel("Distance (mm)")
            ax.legend(loc='upper right', fontsize='small')
            set_refined_grid(ax)

        ax2.set_xlabel("Time (s)")
        ax1.set_title(f"Refined Threshold Crossing Analysis\nFile: {os.path.basename(file_path)}")
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    plot_with_enhanced_labels()