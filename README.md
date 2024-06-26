
## Установка и запуск

### 1. Установка предварительных пакетов

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```
### 2. Клонирование репозитория
```bash
git clone https://github.com/your_username/your_project.git
cd your_project
```

### 3. Создание и активация виртуального окружения
```bash
python3 -m venv llama_env
source llama_env/bin/activate
```

### 4. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 5. Запуск обучения модели
```bash
accelerate launch train_model.py
```

### 6. Запуск скрипта для проверки
```bash
python classify_text.py
```

