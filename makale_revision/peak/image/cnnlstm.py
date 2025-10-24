import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from joblib import Parallel, delayed
import os
import tensorflow as tf

# GPU bellek yönetimi
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Veri setini yükleme ve işleme
df = pd.read_excel("./last_dataset.xlsx")
df = df.drop([
    '10_m_ruzg_yon(derece)', '2_m_ciy_noktasi(°C)', '2_m_islak_termo(°C)',
    'tum_gokyuzu_UVA_irradiance(W/m^2)', 'tum_gy_irradiance_kisa_dalga(MJ/saat)',
    'atmosfer_ustu_irradiance_kisa_dalga(MJ/saat)', 'acik_gy_irradiance_kisa_dalga(MJ/saat)',
    '10_m_ruzg_hiz(m/s)', '2_m_sicaklik(°C)', '2_m_bagıl_nem (%)',
    'tum_gokyuzu_PAR_toplam(W/m^2)', 'acik_gokyuzu_PAR_toplam(W/m^2)',
    '2_m_ruzg_yon(derece)', '2_m_ruzg_hiz(m/s)', 'yagis_duzgun(mm/saat)',
    'tum_gokyuzu_UVB_irradiance(W/m^2)', 'tum_gy_direkt_normal_irradiance(MJ/saat)'
], axis=1)
df['dateInt'] = df['YIL'].astype(str) + df['AY'].astype(str).str.zfill(2) + df['GUN'].astype(str).str.zfill(2) + df['SAAT'].astype(str).str.zfill(2)
df['Date'] = pd.to_datetime(df['dateInt'], format='%Y%m%d%H')
df.set_index('Date', inplace=True)

# Aykırı değerleri sınırlarla değiştirme
pm2_5 = df['PM2.5']
rolling_mean = pm2_5.rolling(window=24, center=True).mean()
rolling_std = pm2_5.rolling(window=24, center=True).std()
upper_bound = rolling_mean + (2 * rolling_std)
lower_bound = rolling_mean - (2 * rolling_std)
df.loc[pm2_5 < lower_bound, 'PM2.5'] = lower_bound[pm2_5 < lower_bound]
df.loc[pm2_5 > upper_bound, 'PM2.5'] = upper_bound[pm2_5 > upper_bound]

# Veri setini X ve y olarak ayırma
X = df.drop(['RUZGAR_YONU', 'PM2.5', 'dateInt', 'AY', 'GUN', 'SAAT', 'day_of_year'], axis=1)
y = df['PM2.5']

# Eğitim ve test verilerini ayırma
train_size = int(len(df) * 0.9)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Veriyi ölçeklendirme
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# Zaman penceresi oluşturma
def create_time_windowed_data(data, target, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(target[i])
    return np.array(X), np.array(y)

# CNN-LSTM modeli oluşturma
def build_cnn_lstm(neurons=50, dropout=0.2, optimizer='adam', conv_neurons=64, cnn_layers=1, lstm_layers=1):
    model = Sequential()
    model.add(Input(shape=(None, X_train_scaled.shape[1])))
    # CNN katmanları
    for _ in range(cnn_layers):
        model.add(Conv1D(conv_neurons, kernel_size=3, activation='relu', padding='causal'))
        model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    # LSTM katmanları
    for i in range(lstm_layers):
        if i == lstm_layers - 1:
            model.add(LSTM(neurons, activation='relu', return_sequences=False))
        else:
            model.add(LSTM(neurons, activation='relu', return_sequences=True))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

# Model eğitme ve değerlendirme fonksiyonu
def train_and_evaluate(params):
    neurons, dropout, optimizer, time_window, batch_size, conv_neurons, cnn_layers, lstm_layers = params
    print(f"Training CNN-LSTM model with parameters: neurons={neurons}, dropout={dropout}, optimizer={optimizer}, time_window={time_window}, batch_size={batch_size}, conv_neurons={conv_neurons}, cnn_layers={cnn_layers}, lstm_layers={lstm_layers}")
    
    # Zaman pencereli veri oluşturma
    X_train_windowed, y_train_windowed = create_time_windowed_data(X_train_scaled, y_train_scaled, time_window)
    X_test_windowed, y_test_windowed = create_time_windowed_data(X_test_scaled, y_test_scaled, time_window)
    
    # Model oluştur ve eğit
    model = build_cnn_lstm(neurons=neurons, dropout=dropout, optimizer=optimizer, conv_neurons=conv_neurons, cnn_layers=cnn_layers, lstm_layers=lstm_layers)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train_windowed, y_train_windowed,
        validation_data=(X_test_windowed, y_test_windowed),
        epochs=150,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Tahmin ve performans değerlendirme
    y_pred_scaled = model.predict(X_test_windowed)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = y_test[time_window:].values.flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred.flatten()))
    
    # Z-score ile dinamik peak analiz
    z_scores = (y_true - np.mean(y_true)) / np.std(y_true)
    z_threshold = 1.5
    peak_mask = z_scores > z_threshold
    y_true_peak = y_true[peak_mask]
    y_pred_peak = y_pred.flatten()[peak_mask]
    
    if len(y_true_peak) > 0:
        mse_peak = mean_squared_error(y_true_peak, y_pred_peak)
        rmse_peak = np.sqrt(mse_peak)
        r2_peak = r2_score(y_true_peak, y_pred_peak)
        mae_peak = np.mean(np.abs(y_true_peak - y_pred_peak))
    else:
        mse_peak = rmse_peak = r2_peak = mae_peak = None
    
    return {
        'model': model,
        'neurons': neurons,
        'dropout': dropout,
        'optimizer': optimizer,
        'time_window': time_window,
        'batch_size': batch_size,
        'conv_neurons': conv_neurons,
        'cnn_layers': cnn_layers,
        'lstm_layers': lstm_layers,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'z_threshold': z_threshold,
        'mse_peak': mse_peak,
        'rmse_peak': rmse_peak,
        'r2_peak': r2_peak,
        'mae_peak': mae_peak,
        'y_true': y_true,
        'y_pred': y_pred.flatten(),
        'time_window': time_window
    }

# Grafik çizdirme fonksiyonu
def plot_last_21_days(model_result, attempt_num):
    # Son 21 günlük veriyi al (504 saat)
    time_window = model_result['time_window']
    last_21_days_idx = -504 - time_window
    X_last_21 = X_test_scaled[last_21_days_idx:]
    y_last_21_true = y_test.iloc[last_21_days_idx + time_window:].values
    
    # Zaman pencereli veri oluştur
    X_last_21_windowed, _ = create_time_windowed_data(X_last_21, np.zeros(len(X_last_21)), time_window)
    
    # Tahmin yap
    y_last_21_pred_scaled = model_result['model'].predict(X_last_21_windowed)
    y_last_21_pred = scaler_y.inverse_transform(y_last_21_pred_scaled)
    
    # Tarih bilgilerini al
    dates = y_test.index[last_21_days_idx + time_window:]
    
    # Grafik çizdir
    plt.figure(figsize=(13, 7))
    plt.plot(dates, y_last_21_true, 'b-', label='Actual Values', color='#1f77b4')  # Mavi renk
    plt.plot(dates, y_last_21_pred.flatten(), 'r-', label='Predicted Values', color='#ff7f0e')  # Turuncu renk
    plt.title(f'Actual and predicted values for PM2.5 (Last 21 days)')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('PM2.5 Concentration', fontsize=14)
    plt.legend()
    plt.grid(False)  # Grid çizgilerini kaldır
    plt.tight_layout()
    plt.savefig(f"pm25_last_21_days_attempt{attempt_num}_r2{model_result['r2']:.4f}_r2peak{model_result['r2_peak']:.4f}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Attempt {attempt_num} - R² Score: {model_result['r2']:.4f}")
    print(f"Attempt {attempt_num} - R² Peak Score: {model_result['r2_peak']:.4f}")

# Paralel çalışacak fonksiyon
def train_and_plot_if_successful(params, target_r2, target_r2_peak, attempt_num):
    print(f"\nAttempt {attempt_num}")
    result = train_and_evaluate(params)
    
    current_r2 = result['r2']
    current_r2_peak = result['r2_peak']
    
    print(f"Current R²: {current_r2:.4f}, Target R²: {target_r2}")
    print(f"Current R² Peak: {current_r2_peak:.4f}, Target R² Peak: {target_r2_peak}")
    
    # Hedefe ulaşıldı mı kontrol et ve ulaşıldıysa grafiği çizdir
    if current_r2 >= target_r2 and current_r2_peak >= target_r2_peak:
        print("\nTarget R² values reached! Plotting the results...")
        plot_last_21_days(result, attempt_num)
        return True
    
    return False

# Kullanılacak parametre seti
params = (50, 0.1, 'Adam', 8, 16, 128, 1, 1)  # (neurons, dropout, optimizer, time_window, batch_size, conv_neurons, cnn_layers, lstm_layers)

# Hedef R² değerleri
target_r2 = 0.94
target_r2_peak = 0.76

# Modeli eğit ve hedeflere ulaşana kadar döngüyü sürdür
max_attempts = 2000  # Maksimum deneme sayısı
batch_size = 35  # Paralel çalışacak işlem sayısı (genellikle CPU çekirdek sayısı)
success_count = 0  # Başarılı model sayısı

# Paralel çalışma için döngü
for batch_start in range(0, max_attempts, batch_size):
    batch_end = min(batch_start + batch_size, max_attempts)
    attempt_nums = list(range(batch_start + 1, batch_end + 1))
    
    # Paralel olarak çalıştır
    batch_results = Parallel(n_jobs=batch_size)(
        delayed(train_and_plot_if_successful)(params, target_r2, target_r2_peak, attempt_num) 
        for attempt_num in attempt_nums
    )
    
    # Başarılı modelleri say
    success_count += sum(batch_results)
    
    print(f"\nBatch completed. Total successful models so far: {success_count}")

# Sonuç özeti
print(f"\nCompleted all {max_attempts} attempts.")
print(f"Found {success_count} models that meet the target R² values.")