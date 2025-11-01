import os
import pandas as pd
import io
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI

# Загружаем переменные из .env файла
load_dotenv()

# Инициализация OpenRouter клиента
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Список доступных бесплатных моделей в порядке приоритета
MODELS = [
    "meta-llama/llama-2-70b-chat",           # Llama 2 70B
    "meta-llama/llama-3-70b-instruct",       # Llama 3 70B
    "mistralai/mistral-7b-instruct",         # Mistral 7B
    "meta-llama/llama-2-13b-chat",           # Llama 2 13B
    "nousresearch/nous-hermes-2-mixtral-8x7b-dpo",  # Nous Hermes
]

current_model_index = 0

# Промпт для анализа маркетплейсов
SYSTEM_PROMPT = """Ты профессиональный аналитик маркетплейсов с опытом работы с Ozon, Wildberries, Яндекс.Маркет и другими платформами.

Твоя задача анализировать выгрузки данных продавцов и давать конкретные, практические рекомендации.

Когда пользователь предоставляет файлы с данными маркетплейса, следуй этому алгоритму:

1. ПОДТВЕРЖДЕНИЕ ПОЛУЧЕНИЯ
Кратко подтверди, что ты получил файлы, понял какой период они охватывают и какие данные содержат.

2. АНАЛИТИЧЕСКИЙ ОТЧЕТ (формат)

🚀 САММАРИ (Главное за 30 секунд)
- 3-5 ключевых выводов: что было хорошо, что плохо, на что срочно обратить внимание
- Пример: "Выручка +15%, но прибыль упала из-за логистики. Товар X - хит, товар Y съедает склад"

💡 КЛЮЧЕВЫЕ РЕКОМЕНДАЦИИ (Приоритизированные)
- 3-5 самых важных действий для выполнения прямо сейчас
- Пример: "1. Дозаказать товар X (остаток на 5 дней). 2. Поднять цену на товар Z на 10%"

📊 ДЕТАЛЬНЫЙ РАЗБОР

Финансовые показатели:
- Оборот (Выручка): общая сумма заказов
- Комиссии и расходы: что отдали маркетплейсу
- Чистая прибыль и маржинальность: реальный доход
- Динамика: изменения vs предыдущий период

ABC-анализ товаров:
- Группа A (Локомотивы): Топ-5 товаров, дающих 80% прибыли
- Группа B (Середняки): стабильные товары
- Группа C (Балласт): непроходимые товары, рекомендации по действиям

Анализ запасов:
- Out-of-Stock риски: какие товары закончатся в ближайшее время
- "Замороженные деньги": товары с низкой оборачиваемостью

Проблемные зоны:
- Возвраты: % возвратов, какие товары возвращают часто
- "Красные флаги": любые аномалии (падение продаж, рост комиссий, штрафы)

3. СТИЛЬ КОММУНИКАЦИИ
- Пиши простым "человеческим" языком, как бизнес-партнер
- Объясняй сложные термины просто
- Не бойся плохих новостей - честность важна
- Будь проактивен: замечай возможности и угрозы

4. ДОП. ЗАПРОСЫ
Если пользователь просит что-то типа:
- "Почему упали продажи по товару X"
- "Сравни две рекламные кампании"
- "Выгодна ли эта акция"
- Отвечай конкретно, с расчетами и выводами

Если в данных не хватает информации для полного анализа (например, себестоимость), скажи об этом явно и попроси недостающие данные."""

# Хранилище контекста диалога для каждого пользователя
user_conversations = {}

async def call_ai_with_fallback(messages):
    """Функция для вызова AI с автоматическим переключением моделей при ошибке"""
    global current_model_index
    
    for attempt in range(len(MODELS)):
        model = MODELS[current_model_index]
        try:
            messages_with_system = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
            response = client.chat.completions.create(
                model=model,
                max_tokens=4000,
                messages=messages_with_system,
                temperature=0.7
            )
            return response.choices[0].message.content, model
        except Exception as e:
            print(f"⚠️ Ошибка с моделью {model}: {str(e)}")
            current_model_index = (current_model_index + 1) % len(MODELS)
            
            if attempt == len(MODELS) - 1:
                raise Exception(f"❌ Все модели недоступны. Последняя ошибка: {str(e)}")
    
    return None, None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /start"""
    user_id = update.effective_user.id
    user_conversations[user_id] = []
    
    await update.message.reply_text(
        "👋 Привет! Я ваш аналитик маркетплейсов.\n\n"
        "Я помогу вам разобраться в выгрузках данных с Ozon, Wildberries, Яндекс.Маркета и других платформ.\n\n"
        "Просто отправьте мне:\n"
        "📁 Excel или CSV файлы с данными маркетплейса\n"
        "❓ Или напишите вопрос по данным, которые вы ранее отправили\n\n"
        "Я проанализирую всё и дам конкретные рекомендации!"
    )

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка загруженных файлов"""
    user_id = update.effective_user.id
    
    if user_id not in user_conversations:
        user_conversations[user_id] = []
    
    try:
        # Получаем файл
        file = await update.message.document.get_file()
        file_bytes = await file.download_as_bytearray()
        
        # Определяем тип файла
        filename = update.message.document.file_name
        
        # Читаем файл в зависимости от расширения
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(io.BytesIO(file_bytes))
        elif filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            await update.message.reply_text("❌ Поддерживаются только файлы Excel (.xlsx, .xls) и CSV")
            return
        
        # Преобразуем данные в текстовый формат
        data_preview = f"Файл: {filename}\n\n"
        data_preview += f"Размер: {len(df)} строк, {len(df.columns)} колонок\n\n"
        data_preview += "Колонки: " + ", ".join(df.columns.tolist()) + "\n\n"
        data_preview += "Данные:\n" + df.to_string()
        
        # Отправляем сообщение пользователю
        await update.message.reply_text("⏳ Анализирую данные... (это может занять некоторое время)")
        
        # Добавляем сообщение пользователя в историю
        user_conversations[user_id].append({
            "role": "user",
            "content": f"Вот мои данные с маркетплейса:\n\n{data_preview}"
        })
        
        # Вызываем AI с автоматическим переключением
        assistant_message, used_model = await call_ai_with_fallback(user_conversations[user_id])
        
        # Сохраняем ответ в историю
        user_conversations[user_id].append({
            "role": "assistant",
            "content": assistant_message
        })
        
        # Отправляем ответ пользователю частями (по 4096 символов)
        for i in range(0, len(assistant_message), 4096):
            await update.message.reply_text(assistant_message[i:i+4096])
        
        print(f"✅ Анализ выполнен моделью: {used_model}")
            
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка при обработке файла: {str(e)}")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка текстовых сообщений (вопросов)"""
    user_id = update.effective_user.id
    user_text = update.message.text
    
    if user_id not in user_conversations:
        user_conversations[user_id] = []
    
    # Добавляем сообщение в историю
    user_conversations[user_id].append({
        "role": "user",
        "content": user_text
    })
    
    try:
        await update.message.reply_text("⏳ Ищу ответ...")
        
        # Вызываем AI с автоматическим переключением
        assistant_message, used_model = await call_ai_with_fallback(user_conversations[user_id])
        
        # Сохраняем ответ в историю
        user_conversations[user_id].append({
            "role": "assistant",
            "content": assistant_message
        })
        
        # Отправляем ответ пользователю частями
        for i in range(0, len(assistant_message), 4096):
            await update.message.reply_text(assistant_message[i:i+4096])
        
        print(f"✅ Ответ от модели: {used_model}")
            
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {str(e)}\nПроверьте OPENROUTER_API_KEY в файле .env")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /reset для очистки истории"""
    user_id = update.effective_user.id
    user_conversations[user_id] = []
    await update.message.reply_text("🔄 История диалога очищена. Готов к новому анализу!")

async def models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /models для просмотра доступных моделей"""
    models_list = "\n".join([f"• {m}" for m in MODELS])
    await update.message.reply_text(f"📋 Доступные модели:\n\n{models_list}")

def main() -> None:
    """Запуск бота"""
    # Получаем токен из переменной окружения
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("❌ Переменная окружения TELEGRAM_BOT_TOKEN не установлена! Проверьте файл .env")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("❌ Переменная окружения OPENROUTER_API_KEY не установлена! Проверьте файл .env")
    
    # Создаем приложение
    application = Application.builder().token(token).build()
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(CommandHandler("models", models))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    # Запускаем бота
    print("🤖 Бот запущен с OpenRouter!")
    print(f"✅ Доступно {len(MODELS)} моделей с автоматическим переключением")
    print("📋 Команды: /start, /reset, /models")
    application.run_polling()

if __name__ == '__main__':
    main()