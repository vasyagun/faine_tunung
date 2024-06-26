import re
import emoji
import pymorphy2
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import json

# Загрузка конфигурации
with open('config.json') as config_file:
    config = json.load(config_file)

model_name = config['model_name']

# Инициализация анализатора для лемматизации
ma = pymorphy2.MorphAnalyzer()

# Функция для очистки и нормализации текста
def clean_text_for_spam_check(text):
    text = emoji.demojize(text, delimiters=("[", "]"))
    text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
    text = text.lower()
    text = re.sub('\-\s\r\n\s{1,}|\-\с\r\n|\р\n', '', text)  # Удаляем новые строки и разрывы строк
    text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\\|\с{2,}|-', ' ', text)  # Удаляем символы
    words = text.split()
    normalized_words = [ma.parse(word)[0].normal_form for word in words if len(word) > 1]
    return " ".join(normalized_words)

# Загрузка токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Создание пайплайна для классификации
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Функция для классификации текста
def classify_text(text):
    cleaned_text = clean_text_for_spam_check(text)
    return classifier(cleaned_text)

# Пример использования
if __name__ == "__main__":
    input_text = input("Введите текст для классификации: ")
    result = classify_text(input_text)
    print(result)
