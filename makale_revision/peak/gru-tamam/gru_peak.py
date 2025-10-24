import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from itertools import product
from joblib import Parallel, delayed

# Veri setini yükleme ve işleme
df = pd.read_excel("../last_dataset.xlsx")
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

# GRU modeli oluşturma
def build_gru(neurons=50, dropout=0.2, optimizer='adam', h_layers=2):
    model = Sequential()
    model.add(Input(shape=(None, X_train_scaled.shape[1])))
    for i in range(h_layers):
        return_seq = i != h_layers - 1
        model.add(GRU(neurons, activation='relu', return_sequences=return_seq))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

# Model eğitme ve değerlendirme fonksiyonu
def train_and_evaluate(params):
    neurons, dropout, h_layers, optimizer, time_window, batch_size = params
    print(f"Training GRU: neurons={neurons}, dropout={dropout}, h_layers={h_layers}, optimizer={optimizer}, time_window={time_window}, batch_size={batch_size}")
    
    # Zaman pencereli veri oluştur
    X_train_win, y_train_win = create_time_windowed_data(X_train_scaled, y_train_scaled, time_window)
    X_test_win, y_test_win = create_time_windowed_data(X_test_scaled, y_test_scaled, time_window)

    # Model eğitimi
    model = build_gru(neurons, dropout, optimizer, h_layers)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        X_train_win, y_train_win,
        validation_data=(X_test_win, y_test_win),
        epochs=150,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )

    # Tahmin ve performans
    y_pred_scaled = model.predict(X_test_win)
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

    # Sonuçları dosyaya yaz
    with open("gru_zscore_peak_results.txt", "a") as f:
        f.write(str({
            'neurons': neurons,
            'dropout': dropout,
            'h_layers': h_layers,
            'optimizer': optimizer,
            'time_window': time_window,
            'batch_size': batch_size,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'z_threshold': z_threshold,
            'mse_peak': mse_peak,
            'rmse_peak': rmse_peak,
            'r2_peak': r2_peak,
            'mae_peak': mae_peak
        }) + "\n")

    return {
        'neurons': neurons,
        'dropout': dropout,
        'h_layers': h_layers,
        'optimizer': optimizer,
        'time_window': time_window,
        'batch_size': batch_size,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'z_threshold': z_threshold,
        'mse_peak': mse_peak,
        'rmse_peak': rmse_peak,
        'r2_peak': r2_peak,
        'mae_peak': mae_peak
    }

# Hiperparametre kombinasyonları
neurons_list = [50,100,150,200] # 4
dropout_list = [0.1, 0.2] # 2
h_layers_list = [2, 3, 4] # 3
optimizer_list = ['adam', 'rmsprop'] # 2
time_window_sizes = [8, 24, 72] # 3
batch_size_list = [16, 32, 64] # 3

param_combinations = list(product(
    neurons_list, dropout_list, h_layers_list,
    optimizer_list, time_window_sizes, batch_size_list
))

# Modelleri paralel olarak eğit ve değerlendir
results = Parallel(n_jobs=10)(
    delayed(train_and_evaluate)(params) for params in param_combinations
)

print("Z-score tabanlı peak analizi tamamlandı. Sonuçlar 'gru_zscore_peak_results.txt' dosyasına kaydedildi.")
