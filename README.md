
![matrix](https://github.com/user-attachments/assets/d3a133ff-488c-419d-bb2c-f9ce7d25ae68)

# Instrukcja
##### Odpalamy program, główny plik programu "find_emotion.py"

![image](https://github.com/user-attachments/assets/888e822e-1d4e-4043-b602-93eae433cbc5)

##### Następnie odpali się nasze GUI, które czeka na otawrcie pliku z danymy EEG w CSV

![image](https://github.com/user-attachments/assets/8f7ac213-9bc4-4ce3-a6b8-aa45767d16b1)

Zwrot programu: 
- Przewidywana emocja, która jest automatycznie ładowana do modelu LLM
- Wykres punktowy emocji
- Model LLM, automatycznie generuje odpowiedz na pierwsze pytanie zgodnie z emocją (pierwsze pytanie na szytwno w kodzie)

![image](https://github.com/user-attachments/assets/91fd5b3d-a37d-4d81-8d54-b0b817dcd737)
![image](https://github.com/user-attachments/assets/b5956c3d-138e-4c12-9811-04bd811093a0)


# RAG:
#### *rag_utils.py*
- Plik do budowania własnej bazy wiedzy
- *knowledge.txt* -> Podajemy swoje informacje które chcemy aby LLM brał pod uwagę w pierwszej kolejności
- Aktualizacja bazy -> nalezy usunąć cały folder *faiss_index/* - zaktualizować plik *knowledge.txt* następnie odpalić *rag_utils.py* 

*Główny program w pierwszej kolejności korzysta z gotowej bazy wiedzy RAG. Jeżeli odpowiedz nie będzie miała sensu w aktualnym kontekscie to skorzysta z modelu LLM*

# Plik Jupyter:
#### *train_emotions.ipynb*
- Jest pobrany z Kaggle ->> gotowiec, zmieniono tylko niektóre biblioteki na aktualne oraz scieżki do plików treingowych pobranych również z Kaggle.
- Model KERAS oraz scaler jest zapisywany po wytrenowaniu na koncu train_emotions.ipynb. Aby móc z korzystać później z wytrenowanego gotowca.

# Model językowy:
- #### Ollama - SpeakLeash/bielik-11b-v2.3-instruct-imatrix:IQ1_M (mniejszy) / SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M (większy)
- #### inicjacja - *find_emotions.py*
Należy pobrać Ollamę z [głównej strony](https://ollama.com/):

![image](https://github.com/user-attachments/assets/740bb195-53bc-4fd2-a63a-7f7f39148c2a)

I mieć ją odpaloną jako usługa w trakcie korzystania z programu.
Dodatkowo należy pobrać odpowiedni model:
- ``` ollama pull SpeakLeash/bielik-11b-v2.3-instruct-imatrix:IQ1_M ```
- ``` ollama pull SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M ```
