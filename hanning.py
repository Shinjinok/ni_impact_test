import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 준비 (123개의 샘플)
n = 123
x = np.linspace(0, 1, n)
# 약간의 진동이 있는 랜덤 신호 생성
data = np.sin(2 * np.pi * 5 * x) + np.random.normal(0, 0.2, n)

# 2. 해닝 창 생성
window = np.hanning(n)

# 3. 윈도우 적용 (Element-wise multiplication)
processed_data = data * window

# 4. 시각화
plt.figure(figsize=(12, 6))

# 원본 데이터 플롯
plt.subplot(1, 2, 1)
plt.plot(data, color='gray', alpha=0.5, label='Original')
plt.title("Original Signal")
plt.grid(True)

# 해닝 창이 적용된 데이터 플롯
plt.subplot(1, 2, 2)
plt.plot(processed_data, color='blue', label='Windowed')
plt.plot(window, 'r--', label='Hanning Shape', alpha=0.3) # 해닝 창의 형태도 같이 표시
plt.title("Signal with Hanning Window")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()