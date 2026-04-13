# pj15_rss_news_oil_gaz_1855

Биржевая торговля на Мосбирже по новостной стратегии на основе эмбеддингов.

## Описание

Система собирает RSS-новости по нефти и газу (Investing, 1Prime, Interfax), генерирует векторные представления (эмбеддинги) через локальную Ollama, находит исторически похожие дни по косинусному сходству и предсказывает направление цены фьючерса на следующую торговую сессию. Настроено для фьючерса на индекс РТС.

## Пайплайн

```
RSS-новости (SQLite) → markdown-файлы → эмбеддинги (Ollama)
    → поиск похожих дней → симуляция торговли → анализ стратегии
```

Скрипты запускаются последовательно, каждый читает конфигурацию из `rts/settings.yaml`:

1. **`download_minutes_to_db.py`** — загрузка минутных свечей с MOEX ISS API в SQLite
2. **`convert_minutes_to_days.py`** — агрегация в дневные бары (сессия 21:00–20:59:59 МСК)
3. **`create_markdown_files.py`** — формирование .md файлов из новостей (фильтр: нефть/газ)
4. **`create_embedding.py`** — генерация эмбеддингов через Ollama (модели: embeddinggemma, bge-m3, qwen3-embedding)
5. **`simulate_trade.py`** — бэктест: поиск похожих дней, формирование P/L, подбор оптимального k
6. **`strategy_analysis.py`** — HTML-отчёт (Plotly): equity, просадка, Sharpe, Sortino, Calmar и др.

## Вспомогательные скрипты

- **`check_pkl.py`** — просмотр содержимого кэша эмбеддингов
- **`analyze_explain.py`** — анализ explain-данных, white/black-листы заголовков

## Запуск

```bash
pip install -r requirements.txt

python rts/download_minutes_to_db.py
python rts/convert_minutes_to_days.py
python rts/create_markdown_files.py
python rts/create_embedding.py
python rts/simulate_trade.py
python rts/strategy_analysis.py
```

## Зависимости

- Python 3.10+
- **Ollama** на `localhost:11434` с загруженной моделью эмбеддингов
- Библиотеки из `requirements.txt`
