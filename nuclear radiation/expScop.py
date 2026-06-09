import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal

# יצירת רשימת הקבצים
file_paths = []
for i in range(0, 8):
    file_paths.append("scopData\WaveData23" + str(i) + ".csv")

# --- הגדרות הפילטרים ---
# הכנס לכאן את כל תדרי הרעש שזיהית (למשל 50 ו-150 הרץ)
noise_freqs = [51500.0, 153800.0]
quality_factor = 30.0
filtered_volt = np.array([])

# --- הגדרות מציאת פולסים (זמן מת) ---
# ייתכן שתצטרך לשחק עם הערכים האלו בהתאם לסולם המתח של האות המסונן
PEAK_PROMINENCE = 0.008  # כמה הפולס צריך לבלוט מהסביבה
PEAK_DISTANCE = 10  # מרחק מינימלי (באינדקסים) בין שני פולסים

plt.figure("Volt", figsize=(12, 6))
plt.figure("FFT", figsize=(12, 6))

current_time_offset = 0.0
for file in file_paths:
    try:
        data = np.genfromtxt(file, delimiter=',', skip_header=3)
        time = data[:, 0]
        volt = data[:, 1]

        N = len(time)
        dt = time[1] - time[0]
        fs = 1.0 / dt

        # --- סינון הרעשים בטור ---
        filtered_volt = volt.copy()

        for freq in noise_freqs:
            b, a = signal.iirnotch(freq, quality_factor, fs)
            filtered_volt = signal.filtfilt(b, a, filtered_volt)

        # --- מציאת נקודות המינימום על האות המסונן ---
        # מכפילים במינוס 1 כי אנחנו מחפשים פולסים שיורדים למטה
        inverted_volt = -filtered_volt
        peaks_indices, _ = signal.find_peaks(inverted_volt, prominence=PEAK_PROMINENCE, distance=PEAK_DISTANCE)

        file_name = os.path.basename(file)

        # חישוב והדפסת הזמן המת אם נמצאו מספיק פולסים
        if len(peaks_indices) >= 2:
            peak_times = time[peaks_indices]
            time_differences = np.diff(peak_times)
            dead_time = time_differences[0]
            print(f"[{file_name}] נמצא זמן מת: {dead_time:.6f} שניות")
        else:
            print(f"[{file_name}] לא נמצאו מספיק פולסים לחישוב זמן מת.")

        # --- עיבוד פורייה ---
        fft_values = np.fft.fft(filtered_volt)
        frequencies = np.fft.fftfreq(N, dt)

        amplitudes = (2.0 / N) * np.abs(fft_values)

        positive_freqs = frequencies[:N // 2]
        positive_amps = amplitudes[:N // 2]
        plt.figure("FFT")
        plt.plot(positive_freqs, positive_amps, label=file_name, alpha=0.7)

        # --- שרטוט המתח עם נקודות המינימום ---
        shifted_time = (time - time[0]) + current_time_offset
        plt.figure("Volt")

        # שרטוט הגל עצמו
        plt.plot(shifted_time, filtered_volt, label=file_name)

        # הוספת האיקסים האדומים בנקודות המינימום
        # משתמשים באינדקסים שמצאנו כדי לשלוף את הזמן (המוזז) והמתח המדויקים
        if len(peaks_indices) > 0:
            plt.plot(shifted_time[peaks_indices], filtered_volt[peaks_indices], "rx", markersize=8)

        current_time_offset = shifted_time[-1] + dt

    except Exception as e:
        print(f"Could not process {file}: {e}")

# עיצוב הגרף FFT
# plt.figure("FFT")
# plt.title("Combined Frequency Spectrum (After filter)")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Amplitude (V)")
# plt.legend(loc='upper right')
# plt.grid(True)

# עיצוב גרף המתח
plt.figure("Volt")
plt.title("Oscilloscope Filtered Signal with Detected Minimums")
plt.xlabel("Time [sec]")
plt.ylabel("Voltage [v]")
plt.grid(True)
plt.show()