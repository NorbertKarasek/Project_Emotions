#### Instrukcja
Odpalamy program w konsoli, główny plik programu "find_emotion.py" + wskazujemy na plik .csv z uporządkowanymi danymi EEG według schematu, sample są w \Data\Samples\

### python find_emotion.py D:\Python_Projekty\Project_Emotions\Data\Samples\new_sample_positive.csv

Zwrot programu w konsoli (Przewidywana emocja) + wykres.
CSV przyszłościowo musi dać użytkownik, którą wczytamy zamiast pliku lokalnego aby wytrenowany model (model.keras) zwrócił na emocję i wykres do niej.
CSV wystarczy przekazać do wykonania find_emotion.py ... i zwrot wyników wyświetlić na froncie.

![image](https://github.com/user-attachments/assets/96eb1cc2-bdd4-40f4-95e6-c849f8f0fe8b)

+ Model językowy -> teraz należy jeszcze zaimplementować model LLM który będzie dostawał informację że użytkownik jest np. "NEGATIVE" i ma zwrócić nam tekst co z tym zrobić.

Plik Jupyter:
## train_emotions.ipynb
Jest pobrany z Kaggle ->> gotowiec, zmieniono tylko niektóre biblioteki na aktualne oraz scieżki do plików treingowych pobranych również z Kaggle.
