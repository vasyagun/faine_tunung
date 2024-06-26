import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import pickle
import json

# Загрузка конфигурации
with open('config.json') as config_file:
    config = json.load(config_file)

model_name = config['model_name']
num_epochs = config['num_epochs']

# Загрузка данных из CSV файла
data = pd.read_csv('all_data.csv')
data = data.dropna(subset=['text', 'category'])  # Удаляем строки с пустыми значениями в текстах или категориях

# Преобразование категорий спама
def transform_spam_category(category):
    if category in ['spamMicro', 'spamMid', 'spamShort', 'spamLong']:
        return 'spam'
    return category

data['category'] = data['category'].apply(transform_spam_category)

# Преобразование категорий из текста в числовые значения
label_encoder = LabelEncoder()
data['category'] = label_encoder.fit_transform(data['category'])

# Проверка меток
num_classes = len(label_encoder.classes_)
invalid_labels = data['category'] >= num_classes
if invalid_labels.any():
    print("Found invalid labels: ", data[invalid_labels])
    data = data[~invalid_labels]

# Разделение данных на тренировочные и тестовые наборы
train_texts, val_texts = train_test_split(data, test_size=0.2, random_state=42)

# Инициализация токенизатора
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Токенизация данных
train_encodings = tokenizer(train_texts['text'].tolist(), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts['text'].tolist(), truncation=True, padding=True, max_length=512)

# Создание объектов Dataset
train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask'], 'labels': train_texts['category']})
val_dataset = Dataset.from_dict({'input_ids': val_encodings['input_ids'], 'attention_mask': val_encodings['attention_mask'], 'labels': val_texts['category']})

# Сохранение токенизатора и лейбл энкодера
tokenizer.save_pretrained('./model')
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Создание DatasetDict
dataset_dict = DatasetDict({'train': train_dataset, 'validation': val_dataset})

# Параметры обучения
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",
    eval_steps=50,
    logging_steps=50,
    save_steps=200,
    save_total_limit=1,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    fp16=True,
    dataloader_num_workers=4,
)

# Загрузка модели
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

# Инициализация тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
)

# Обучение модели
trainer.train()

# Сохранение модели
trainer.save_model("./model")
