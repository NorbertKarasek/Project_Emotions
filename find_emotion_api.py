import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import matplotlib.pyplot as plt
# from langchain_ollama import OllamaLLM  # ← USUNIĘTE

# === NOWE: OpenAI ===
from openai import OpenAI
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")  # podmień na swój model

# Jeśli model wymaga tej opcji
tf.keras.config.enable_unsafe_deserialization()

from RAG.rag_utils import load_retriever  # pozostawiamy import; FAISS nie generuje już odpowiedzi

# ===== Pomocnicze: zamiana Twojej historii "### Role: ..." → messages =====
def history_to_messages(history_str: str):
    """
    Konwertuje historię w formacie:
        ### System: ...
        ### User: ...
        ### Assistant: ...
    na listę messages dla OpenAI:
        [{"role": "...", "content": "..."}]
    """
    if not history_str:
        return []

    messages = []
    # Prosta, odporna na \n\n segmentacja po znacznikach "### X:"
    parts = [p.strip() for p in history_str.split("### ") if p.strip()]
    for p in parts:
        if p.startswith("System:"):
            messages.append({"role": "system", "content": p[len("System:"):].strip()})
        elif p.startswith("User:"):
            messages.append({"role": "user", "content": p[len("User:"):].strip()})
        elif p.startswith("Assistant:"):
            messages.append({"role": "assistant", "content": p[len("Assistant:"):].strip()})
        else:
            # fallback – potraktuj jako user
            messages.append({"role": "user", "content": p})
    return messages

# ===== OpenAI chat (zamiast OLLAMA.invoke) =====
class GPTChat:
    def __init__(self, model: str = OPENAI_CHAT_MODEL, temperature: float = 0.2):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def invoke(self, input: str) -> str:
        """
        Zastępstwo dla model.invoke(input=history + '### Assistant:\\n')
        Oczekuje całego 'history' zakończonego '### Assistant:' – tak jak dotychczas.
        """
        # Zamieniamy wejściowy 'input' (cała historia z trailing '### Assistant:\n')
        # na messages dla OpenAI.
        messages = history_to_messages(input)

        # Jeśli ostatnia linia to zapowiedź asystenta bez treści, to nic nie dodajemy.
        # OpenAI i tak odpowie dalszą wiadomością asystenta na podstawie poprzednich wpisów.
        resp = self.client.responses.create(
            model=self.model,
            input=messages,  # Responses API przyjmuje listę wiadomości
            temperature=self.temperature,
        )

        # Pobierz zwykły tekst:
        if getattr(resp, "output_text", None):
            return resp.output_text.strip()

        # Fallback – przeszukaj strukturę output:
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "message":
                parts = []
                for block in getattr(item.message, "content", []) or []:
                    if hasattr(block, "text") and block.text:
                        parts.append(block.text)
                if parts:
                    return "".join(parts).strip()

        return ""


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
    # Zostawiamy kompatybilność z wcześniejszym formatem,
    # ale OpenAI zwróci już czysty tekst.
    if "### Assistant:" in resp:
        return resp.split("### Assistant:")[-1].strip()
    return resp.strip()


def create_gui():
    """Tworzy GUI do analizy emocji na podstawie EEG."""
    # Inicjalizacja modelu i historii
    # model = OllamaLLM(model="SpeakLeash/bielik-11b-v2.3-instruct-imatrix:IQ1_M")  # ← USUNIĘTE
    model = GPTChat(model=OPENAI_CHAT_MODEL, temperature=0.2)

    retriever = load_retriever("RAG/faiss_index")  # pozostawione, ale nie generuje już odpowiedzi
    history = "### System: You are a helpful assistant.\n"

    # ====== (FAISS→GPT) – stara funkcja oceny FAISS usunięta ======
    # def is_faiss_answer_valid_llm(...):  # ← usunięte

    # ====== (FAISS→GPT) – nie używamy FAISS do generowania odpowiedzi ======
    def get_faiss_answer(user_prompt: str):
        """
        Zgodnie z prośbą: odpowiedzi FAISS NIE są wyświetlane.
        Zostawiamy retriever (jeśli chcesz potem użyć go do RAG),
        ale ta funkcja już nie zwraca treści do GUI.
        """
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
        print(f"Przewidywana emocja: {label}")
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

        # --- Zamiast FAISS → od razu GPT ---
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

        # --- Zamiast FAISS → od razu GPT ---
        full_resp = model.invoke(input=history + "### Assistant:\n")
        resp = sanitize_response(full_resp)
        history += f"### Assistant: {resp}\n"
        insert_speaker("Chatbot: ", resp)

        entry.delete(0, tk.END)

    tk.Button(root, text="Otwórz plik", command=analyze_file).pack(pady=5)
    send_button.config(command=send_message)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
