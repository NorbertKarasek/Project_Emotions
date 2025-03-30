import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt

# Włączamy niebezpieczną deserializację, jeśli to konieczne
tf.keras.config.enable_unsafe_deserialization()


def extract_features_from_csv(filepath, desired_features=2548):
    """
    Wczytuje plik CSV z surowymi danymi EEG, gdzie:
      - Nagłówki odpowiadają nazwom elektrod (np. Fp1, AF3, ... O2)
      - Każdy wiersz to kolejny punkt czasowy.

    Dla każdego kanału:
      - Oblicza FFT (transformata Fouriera) na całym szeregu czasowym,
      - Pobiera część rzeczywistą wyników (zachowując informację o znaku),
      - Wybiera pierwsze 'coeffs_per_channel' współczynników.

    Łączy wyniki ze wszystkich kanałów w jeden wektor o długości 'desired_features'.
    Jeśli wektor jest krótszy – dopełnia zerami, a jeśli dłuższy – przycina.
    """
    # Wczytanie danych – upewnij się, że plik ma poprawny format
    if not os.path.exists(filepath):
        print(f"Plik {filepath} nie istnieje!")
        sys.exit(1)

    df = pd.read_csv(filepath)
    # Upewnij się, że nie ma zbędnych kolumn (np. pustej kolumny na końcu)
    df = df.dropna(axis=1, how='all')

    # Zamieniamy dane na tablicę NumPy – shape: (num_timepoints, num_channels)
    raw_data = df.values
    num_channels = raw_data.shape[1]

    # Obliczamy liczbę współczynników FFT na kanał
    coeffs_per_channel = desired_features // num_channels
    features = []

    for i in range(num_channels):
        channel_data = raw_data[:, i]
        # Obliczenie FFT dla całego szeregu czasowego danego kanału
        fft_vals = np.fft.fft(channel_data)
        # Pobranie części rzeczywistej (zachowujemy informację o znaku)
        fft_real = np.real(fft_vals)
        # Wybieramy pierwsze coeffs_per_channel współczynników
        selected_coeffs = fft_real[:coeffs_per_channel]
        features.append(selected_coeffs)

    # Łączymy cechy ze wszystkich kanałów w jeden wektor
    feature_vector = np.concatenate(features)

    # Dopasowujemy długość wektora do desired_features
    if len(feature_vector) < desired_features:
        feature_vector = np.pad(feature_vector, (0, desired_features - len(feature_vector)))
    elif len(feature_vector) > desired_features:
        feature_vector = feature_vector[:desired_features]

    return feature_vector


def main():
    # Sprawdzenie, czy ścieżka do pliku CSV została podana jako argument
    if len(sys.argv) < 2:
        print("Użycie: python find_emotion.py <ścieżka_do_pliku_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Ekstrakcja cech z pliku CSV (pipeline analogiczny do treningu)
    print("Przetwarzanie danych EEG z pliku:", csv_path)
    features = extract_features_from_csv(csv_path, desired_features=2548)
    features = features.reshape(1, -1)
    print("Kształt wektora cech po ekstrakcji:", features.shape)

    # Wczytanie wytrenowanego modelu i skalera
    model_path = 'model.keras'
    scaler_path = 'scaler.pkl'

    if not os.path.exists(model_path):
        print(f"Model nie został znaleziony pod ścieżką {model_path}")
        sys.exit(1)
    if not os.path.exists(scaler_path):
        print(f"Scaler nie został znaleziony pod ścieżką {scaler_path}")
        sys.exit(1)

    print("Wczytywanie modelu i skalera...")
    model = tf.keras.models.load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Skalowanie cech
    features_scaled = scaler.transform(features)

    # Predykcja emocji
    pred = model.predict(features_scaled)
    predicted_class = np.argmax(pred, axis=1)

    # Mapowanie indeksów na etykiety emocji
    label_mapping = {0: "NEUTRAL", 1: "POSITIVE", 2: "NEGATIVE"}
    predicted_index = predicted_class[0]
    predicted_label = label_mapping.get(predicted_index, "Unknown")

    print("Przewidywana emocja (indeks klasy):", predicted_index, "(", predicted_label, ")")

    df = pd.read_csv(csv_path)

    # # WYKRES 1: Sygnał EEG – wykres kanału fft_749_b (jeśli istnieje)
    # if 'fft_749_b' in df.columns:
    #     plt.figure(figsize=(10, 4))
    #     plt.plot(df['fft_749_b'])
    #     plt.title(f"Wykres sygnału EEG – kanał {df.columns[0]}")
    #     plt.xlabel("Czas")
    #     plt.ylabel("Amplituda")
    #     plt.tight_layout()
    #     plt.show()
    # else:
    #     print("Kolumna fft_749_b nie została odnaleziona.")

    # WYKRES 2: Wykres FFT część b
    # Wybieramy kolumny zaczynające się na "fft_" i kończące na "_b"
    fft_cols = [col for col in df.columns if col.startswith("fft_") and col.endswith("_b")]
    if fft_cols:
        # Sortujemy kolumny według numeru, zakładając format "fft_<numer>_b"
        fft_cols = sorted(fft_cols, key=lambda x: int(x.split('_')[1]))
        # Pobieramy wartości z pierwszego wiersza dla tych kolumn
        fft_values = df.loc[0, fft_cols].values.astype(float)
        plt.figure(figsize=(12, 6))
        plt.plot(fft_values, marker='o', linestyle='-', color='blue')
        plt.title("Wykres sygnału EEG – FFT część b")
        plt.xlabel("Indeks współczynnika FFT")
        plt.ylabel("Wartość")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("Brak kolumn odpowiadających FFT część b.")


if __name__ == "__main__":
    main()

