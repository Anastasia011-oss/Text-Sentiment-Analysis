import csv
from transformers import pipeline

MODEL_1 = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
MODEL_2 = "nlptown/bert-base-multilingual-uncased-sentiment"

analyzer = None
second_analyzer = None

reviews = [
    "The application works perfectly. I use it every day.",
    "The delivery was late, and the package was damaged.",
    "The product is acceptable, but nothing special.",
    "Excellent customer support and fast service!",
    "The software crashes every time I open it.",
    "I love this product. It is easy to use.",
    "The battery lasts a very long time.",
    "The customer service was rude and unhelpful.",
    "The quality is poor, and I regret buying it.",
    "Everything arrived on time and in perfect condition.",
    "The application is simple and convenient.",
    "I am disappointed with this purchase.",
    "The price is reasonable for the quality.",
    "This is the worst service I have ever received.",
    "The product works exactly as described.",
    "It is okay, but I expected something better.",
    "The instructions are confusing and incomplete.",
    "I would definitely recommend this product.",
    "The website is slow and difficult to navigate.",
    "Fantastic experience! I will buy it again.",
    "The product is neither good nor bad.",
    "I had to wait for hours to get help.",
    "The design is beautiful and modern.",
    "It stopped working after two days.",
    "The service was quick, friendly, and professional.",
    "I thought this product would be better.",
    "Oh great, another update that breaks everything.",
    "The app is not bad, but it could be improved.",
    "I have used this product for several months, "
    "and it continues to work reliably every day.",
    "The delivery was fast, but the item was damaged "
    "and customer support did not help me."
]

def load_first_model():
    global analyzer

    if analyzer is None:
        print("\nЗагрузка первой модели. Подожди...")
        analyzer = pipeline(
            "sentiment-analysis",
            model=MODEL_1
        )

    return analyzer


def load_second_model():
    global second_analyzer

    if second_analyzer is None:
        print("\nЗагрузка второй модели. Подожди...")
        second_analyzer = pipeline(
            "sentiment-analysis",
            model=MODEL_2
        )

    return second_analyzer


def analyze_review(text, model=None):
    if not text.strip():
        print("Ошибка: отзыв не может быть пустым.")
        return None

    if model is None:
        model = load_first_model()

    try:
        result = model(text, truncation=True)[0]

        return {
            "text": text,
            "label": result["label"],
            "confidence": result["score"] * 100
        }

    except Exception as error:
        print("Не удалось проанализировать отзыв:", error)
        return None

def print_result(result, number=None):
    if result is None:
        return

    if number is not None:
        print(f"\nОтзыв №{number}")

    print("Текст:", result["text"])
    print("Тональность:", result["label"])
    print("Уверенность модели:",
          f'{result["confidence"]:.2f}%')


def show_statistics(results, neutral_supported=False):
    positive = 0
    negative = 0
    neutral = 0

    for result in results:
        if result is None:
            continue

        label = result["label"].upper()

        if label == "POSITIVE":
            positive += 1
        elif label == "NEGATIVE":
            negative += 1
        elif label == "NEUTRAL":
            neutral += 1
        elif "STAR" in label:
            # Для модели с оценками от 1 до 5 звёзд.
            stars = int(label.split()[0])

            if stars <= 2:
                negative += 1
            elif stars == 3:
                neutral += 1
            else:
                positive += 1

    analyzed = positive + negative + neutral

    print("\n========== СТАТИСТИКА ==========")
    print("Проанализировано отзывов:", analyzed)
    print("Положительных:", positive)
    print("Отрицательных:", negative)

    if neutral_supported:
        print("Нейтральных:", neutral)
    else:
        print("Нейтральных: отдельный класс отсутствует в модели")

    print("================================")


def save_txt(results, filename="results.txt"):
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write("РЕЗУЛЬТАТЫ АНАЛИЗА ТОНАЛЬНОСТИ\n")
            file.write(f"Модель: {MODEL_1}\n\n")

            for number, result in enumerate(results, start=1):
                if result is None:
                    continue

                file.write(f"Отзыв №{number}\n")
                file.write(f"Текст: {result['text']}\n")
                file.write(f"Тональность: {result['label']}\n")
                file.write(
                    f"Уверенность: {result['confidence']:.2f}%\n\n"
                )

        print(f"Результаты сохранены в файл {filename}")

    except OSError as error:
        print("Ошибка сохранения TXT:", error)


def save_csv(results, filename="results.csv"):
    try:
        with open(
            filename, "w", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.writer(file)

            writer.writerow([
                "Text",
                "Sentiment",
                "Confidence (%)"
            ])

            for result in results:
                if result is None:
                    continue

                writer.writerow([
                    result["text"],
                    result["label"],
                    f'{result["confidence"]:.2f}'
                ])

        print(f"Результаты сохранены в файл {filename}")

    except OSError as error:
        print("Ошибка сохранения CSV:", error)


def single_review_mode():
    text = input("\nВведи отзыв на английском: ")

    if not text.strip():
        print("Ошибка: ты не ввела текст.")
        return

    result = analyze_review(text)

    if result is not None:
        print_result(result)

        if input("\nСохранить результат? (да/нет): ").lower().strip() == "да":
            save_txt([result])
            save_csv([result])


def batch_mode():
    print("\nНачинаем анализ 30 отзывов...")
    model = load_first_model()
    results = []

    for number, review in enumerate(reviews, start=1):
        result = analyze_review(review, model)

        if result is not None:
            results.append(result)
            print_result(result, number)

    show_statistics(results)

    save_choice = input(
        "\nСохранить результаты в TXT и CSV? (да/нет): "
    ).lower().strip()

    if save_choice == "да":
        save_txt(results)
        save_csv(results)


def custom_reviews_mode():
    print("\nВводи отзывы по одному.")
    print("Чтобы закончить, введи: стоп")

    custom_reviews = []

    while True:
        text = input("\nОтзыв: ").strip()

        if text.lower() == "стоп":
            break

        if not text:
            print("Пустой отзыв пропущен. Попробуй ещё раз.")
            continue

        custom_reviews.append(text)

    if not custom_reviews:
        print("Ты не добавила ни одного отзыва.")
        return

    model = load_first_model()
    results = []

    for number, review in enumerate(custom_reviews, start=1):
        result = analyze_review(review, model)

        if result is not None:
            results.append(result)
            print_result(result, number)

    show_statistics(results)

    if results:
        choice = input(
            "\nСохранить результаты? (да/нет): "
        ).lower().strip()

        if choice == "да":
            save_txt(results)
            save_csv(results)


def compare_models_mode():
    model1 = load_first_model()
    model2 = load_second_model()

    results = []

    print("\nСравнение двух моделей на одинаковых отзывах:")

    for number, review in enumerate(reviews, start=1):
        try:
            result1 = model1(review, truncation=True)[0]
            result2 = model2(review, truncation=True)[0]

            # Вторая модель выдаёт оценку от 1 до 5 звёзд.
            stars = int(result2["label"].split()[0])

            if stars <= 2:
                sentiment2 = "NEGATIVE"
            elif stars == 3:
                sentiment2 = "NEUTRAL"
            else:
                sentiment2 = "POSITIVE"

            print(f"\nОтзыв №{number}: {review}")
            print(
                f"Модель 1: {result1['label']} "
                f"({result1['score'] * 100:.2f}%)"
            )
            print(
                f"Модель 2: {result2['label']} "
                f"({result2['score'] * 100:.2f}%), "
                f"класс: {sentiment2}"
            )

            results.append({
                "text": review,
                "label": result1["label"],
                "confidence": result1["score"] * 100
            })

        except Exception as error:
            print(f"Ошибка при сравнении отзыва №{number}: {error}")

    if results:
        save_txt(results, "comparison_results.txt")
        save_csv(results, "comparison_results.csv")

    print("\nМодель 1:", MODEL_1)
    print("Модель 2:", MODEL_2)
    print("Примечание: модели имеют разные классы и обучающие данные.")


def main():
    while True:
        print("\n========== АНАЛИЗАТОР ОТЗЫВОВ ==========")
        print("1. Анализ одного отзыва")
        print("2. Анализ списка из 30 отзывов")
        print("3. Ввести собственные отзывы")
        print("4. Сравнить две модели Hugging Face")
        print("0. Выход")
        print("========================================")

        choice = input("Выбери пункт меню: ").strip()

        try:
            if choice == "1":
                single_review_mode()

            elif choice == "2":
                batch_mode()

            elif choice == "3":
                custom_reviews_mode()

            elif choice == "4":
                compare_models_mode()

            elif choice == "0":
                print("Программа завершена.")
                break

            else:
                print("Неверный пункт меню. Введи число от 0 до 4.")

        except Exception as error:
            print("Возникла ошибка:", error)
            print("Попробуй повторить действие.")


if __name__ == "__main__":
    main()
