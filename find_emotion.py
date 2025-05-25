import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import matplotlib.pyplot as plt
from langchain_ollama import OllamaLLM

# Jeśli Twój model wymaga tej opcji
tf.keras.config.enable_unsafe_deserialization()

from RAG.rag_utils import load_retriever

def extract_features_from_csv(filepath, desired_features=2548):
    if not os.path.exists(filepath):
        messagebox.showerror("Błąd", f"Plik {filepath} nie istnieje!")
        sys.exit(1)
    df = pd.read_csv(filepath).dropna(axis=1, how='all')
    raw = df.values
    num_ch = raw.shape[1]
    coeffs = desired_features // num_ch
    feats = [np.real(np.fft.fft(raw[:, i]))[:coeffs] for i in range(num_ch)]
    vec = np.concatenate(feats)
    if len(vec) < desired_features:
        vec = np.pad(vec, (0, desired_features - len(vec)))
    else:
        vec = vec[:desired_features]
    return vec.reshape(1, -1), df


def generate_eeg_plot(df):
    fft_cols = [c for c in df.columns if c.startswith("fft_") and c.endswith("_b")]
    if not fft_cols:
        return
    fft_cols.sort(key=lambda x: int(x.split('_')[1]))
    vals = df.loc[0, fft_cols].astype(float).values
    plt.figure(figsize=(8,4))
    plt.plot(vals, marker='o', linestyle='-')
    plt.title("EEG – FFT część b")
    plt.xlabel("Współczynnik")
    plt.ylabel("Amplituda")
    plt.tight_layout()
    plt.show()


def sanitize_response(resp: str) -> str:
    # Wyciąga końcową część odpowiedzi po ostatnim '### Assistant:'
    if "### Assistant:" in resp:
        return resp.split("### Assistant:")[-1].strip()
    return resp.strip()


def create_gui():
    """Tworzy GUI do analizy emocji na podstawie EEG."""
    # Inicjalizacja modelu i historii
    model = OllamaLLM(model="SpeakLeash/bielik-11b-v2.3-instruct-imatrix:IQ1_M")
    retriever = load_retriever("RAG/faiss_index")
    history = "### System: You are a helpful assistant.\n"

    # Funkcja do pobierania odpowiedzi z FAISS
    def is_faiss_answer_valid_llm(model, user_prompt, faiss_answer):
        """Używa LLM do oceny, czy odpowiedź FAISS jest sensowna."""
        check_prompt = (
            f"Czy odpowiedź: '{faiss_answer}' jest merytorycznie zgodna i bezpośrednio odnosi się do pytania: '{user_prompt}'? "
            "Odpowiedz wyłącznie TAK lub NIE. Odpowiedź nie może być nie na temat."
        )
        result = model.invoke(input=check_prompt)
        return "TAK" in result.strip().upper().split()

    def get_faiss_answer(user_prompt: str):
        """Zwraca odpowiedź z FAISS na podstawie zapytania użytkownika, tylko jeśli jest sensowna."""
        if not user_prompt.strip():
            return None
        docs = retriever.invoke(user_prompt)  # zamiast get_relevant_documents
        if docs and docs[0].page_content.strip():
            answer = docs[0].page_content.strip()
            if is_faiss_answer_valid_llm(model, user_prompt, answer):
                return answer
        return None

    # Tworzenie GUI
    root = tk.Tk()
    root.title("EEG Emotion Analyzer")
    root.geometry("600x600")

    tk.Label(root, text="Wybierz plik CSV z danymi EEG:", pady=10).pack()

    text_widget = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=20)
    text_widget.pack(fill=tk.BOTH, expand=True, padx=10)
    text_widget.tag_config("bold", font=("TkDefaultFont", 10, "bold"))

    entry = tk.Entry(root, width=50)
    entry.pack(pady=5)

    send_button = tk.Button(root, text="Wyślij", state=tk.DISABLED)
    send_button.pack(pady=5)

    def insert_speaker(name, msg):
        text_widget.insert(tk.END, name, "bold")
        text_widget.insert(tk.END, msg + "\n\n")
        text_widget.see(tk.END)

    def analyze_file():
        nonlocal history
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return

        feats, df = extract_features_from_csv(path)

        m_path = 'EEG/Model/model.keras'
        s_path = 'EEG/Model/scaler.pkl'
        if not os.path.exists(m_path) or not os.path.exists(s_path):
            messagebox.showerror("Błąd", "Brak pliku modelu lub scalera!")
            return

        net = tf.keras.models.load_model(m_path)
        with open(s_path, 'rb') as f:
            scaler = pickle.load(f)

        scaled = scaler.transform(feats)
        pred = net.predict(scaled)
        idx = int(np.argmax(pred, axis=1)[0])
        label = {0: "NEUTRAL", 1: "POSITIVE", 2: "NEGATIVE"}.get(idx, "Unknown")

        # Pokaż przewidywaną emocję
        insert_speaker("Chatbot: ", f"Przewidywana emocja: {label}")
        generate_eeg_plot(df)

        # Przygotuj automatyczny prompt i wyświetl go jako You:
        user_prompt = {
            "NEUTRAL": "Mój stres jest umiarkowany i potrzebuję wskazówek jak sobie z tym poradzić (w jednym zdaniu)",
            "POSITIVE": "Jestem szczęśliwy, podaj krótką radę jak pozostać w tym stanie (w jednym zdaniu)",
            "NEGATIVE": "Jestem zestresowany i potrzebuję wskazówek jak sobie z tym poradzić (w jednym zdaniu)"
        }[label]
        history += f"### User: {user_prompt}\n"
        insert_speaker("You: ", user_prompt)

        # Najpierw FAISS
        faiss_answer = get_faiss_answer(user_prompt)
        if faiss_answer:
            insert_speaker("Chatbot: ", f"(FAISS) {faiss_answer}")
            send_button.config(state=tk.NORMAL)
            return

        # Dopiero jeśli FAISS nie znalazł, użyj LLM
        full_resp = model.invoke(input=history + "### Assistant:\n")
        resp = sanitize_response(full_resp)
        history += f"### Assistant: {resp}\n"
        insert_speaker("Chatbot: ", resp)
        send_button.config(state=tk.NORMAL)

    def send_message():
        nonlocal history
        user = entry.get().strip()
        if not user:
            return

        # Wyświetl prompt użytkownika
        user_note = f"{user} (w jednym zdaniu)"
        history += f"### User: {user_note}\n"
        insert_speaker("You: ", user_note)

        # Najpierw FAISS
        faiss_answer = get_faiss_answer(user_note)
        if faiss_answer:
            insert_speaker("Chatbot: ", f"(FAISS) {faiss_answer}")
            send_button.config(state=tk.NORMAL)
            return

        # Dopiero jeśli FAISS nie znalazł, użyj LLM
        full_resp = model.invoke(input=history + "### Assistant:\n") # user_note - jeżeli skupiamy się na aktualnym pytaniu | history - jeżeli chcemy uwzględnić całą konwersację
        resp = sanitize_response(full_resp)
        history += f"### Assistant: {resp}\n"
        insert_speaker("Chatbot: ", resp)

        entry.delete(0, tk.END)

    tk.Button(root, text="Otwórz plik", command=analyze_file).pack(pady=5)
    send_button.config(command=send_message)

    root.mainloop()

if __name__ == "__main__":
    create_gui()