import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt
from langchain_ollama import OllamaLLM

# Włączamy niebezpieczną deserializację, jeśli to konieczne
tf.keras.config.enable_unsafe_deserialization()

def extract_features_from_csv(filepath, desired_features=2548):
    """Wczytuje plik CSV z surowymi danymi EEG, gdzie:
       - Nagłówki odpowiadają nazwom elektrod (np. Fp1, AF3, ... O2)
       - Każdy wiersz to kolejny punkt czasowy.
       Dla każdego kanału:
       - Oblicza FFT (transformata Fouriera) na całym szeregu czasowym,
       - Pobiera część rzeczywistą wyników (zachowując informację o znaku),
       - Wybiera pierwsze 'coeffs_per_channel' współczynników.
       Łączy wyniki ze wszystkich kanałów w jeden wektor o długości 'desired_features'.
       Jeśli wektor jest krótszy – dopełnia zerami, a jeśli dłuższy – przycina.
    """
    if not os.path.exists(filepath):
        print(f"Plik {filepath} nie istnieje!")
        sys.exit(1)

    df = pd.read_csv(filepath)
    df = df.dropna(axis=1, how='all')  # Usuwamy kolumny, które są puste

    raw_data = df.values
    num_channels = raw_data.shape[1]

    coeffs_per_channel = desired_features // num_channels
    features = []

    for i in range(num_channels):
        channel_data = raw_data[:, i]
        fft_vals = np.fft.fft(channel_data)
        fft_real = np.real(fft_vals)
        selected_coeffs = fft_real[:coeffs_per_channel]
        features.append(selected_coeffs)

    feature_vector = np.concatenate(features)

    if len(feature_vector) < desired_features:
        feature_vector = np.pad(feature_vector, (0, desired_features - len(feature_vector)))
    elif len(feature_vector) > desired_features:
        feature_vector = feature_vector[:desired_features]

    return feature_vector


# Inicjalizacja modelu LLM (chatbota)
model = OllamaLLM(model="SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M")
history = "### System: You are a helpful assistant.\n"


def chatbot(emotion):
    global history  # Zmieniamy zmienną 'history' na globalną

    emotion_prompts = {
        "NEUTRAL": "Mój stres jest umiarkowany i potrzebuję wskazówek jak sobie z tym poradzić (Udziel krótkiej odpowiedzi w jednym zdaniu)",
        "POSITIVE": "Nie jestem zestresowany, możesz dać mi krótką poradę jak pozostać w tym stanie (Udziel krótkiej odpowiedzi w jednym zdaniu)?",
        "NEGATIVE": "Jestem zestresowany i potrzebuję wskazówek jak sobie z tym poradzić (Udziel krótkiej odpowiedzi w jednym zdaniu)"
    }

    # Użyj odpowiedzi dostosowanej do emocji
    prompt = emotion_prompts.get(emotion, 'How can I assist you today?')
    print(f"Chatbot: {prompt}")

    # Dodajemy zapytanie do historii, ale nie dodajemy odpowiedzi chatbota
    history += f"\n### User:\n{prompt}\n"

    # Wywołanie modelu LLM z emocją jako prompt
    result = model.invoke(input=f"{history}\n### Assistant:\n")

    # Pokazujemy odpowiedź chatbota
    print("Chatbot:", result)
    history += f"### Assistant:\n{result}\n"

    # Pozwalamy użytkownikowi na kontynuowanie rozmowy po odpowiedzi
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        history += f"\n### User:\n{user_input}\n"
        result = model.invoke(input=f"{history}\n### Assistant:\n")

        print("Chatbot:", result)
        history += f"{result}\n"


def generate_eeg_plot(df):
    """Generuje wykres FFT dla danych EEG."""
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


def main():
    if len(sys.argv) < 2:
        print("Użycie: python find_emotion.py <ścieżka_do_pliku_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    print("Przetwarzanie danych EEG z pliku:", csv_path)
    features = extract_features_from_csv(csv_path, desired_features=2548)
    features = features.reshape(1, -1)

    print("Kształt wektora cech po ekstrakcji:", features.shape)

    # Wczytanie wytrenowanego modelu i skalera
    model_path = 'EEG/Model/model.keras'
    scaler_path = 'EEG/Model/scaler.pkl'

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

    features_scaled = scaler.transform(features)

    # Predykcja emocji
    pred = model.predict(features_scaled)
    predicted_class = np.argmax(pred, axis=1)

    label_mapping = {0: "NEUTRAL", 1: "POSITIVE", 2: "NEGATIVE"}
    predicted_index = predicted_class[0]
    predicted_label = label_mapping.get(predicted_index, "Unknown")

    print(f"Przewidywana emocja: {predicted_label} ({predicted_index})")

    # Generowanie wykresu EEG
    df = pd.read_csv(csv_path)
    generate_eeg_plot(df)

    # Rozpocznij rozmowę z chatbotem, uwzględniając przewidywaną emocję
    chatbot(predicted_label)


if __name__ == "__main__":
    main()



