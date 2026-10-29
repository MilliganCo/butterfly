# 4D Butterflies (VisPy)

> Public source preview: see [PUBLICATION.md](PUBLICATION.md) for actual checks, known failures and runtime limits.

Симуляция роя 4D-бабочек в 3D-проекции: AI-поведение, звёзды, светло-серый прозрачный туман, HUD со здоровьем и три режима камеры.

## Описание

Проект визуализирует бабочек, существующих в 4-мерном пространстве (x, y, z, w). Они проецируются в 3D для отображения на экране. Бабочки летят к звёздам, размножаются, теряют здоровье со временем и могут управляться игроком.

## Зависимости

```bash
pip install vispy numpy PySide6
```

Альтернатива PySide6: `pip install PyQt5` (VisPy автоматически выберет доступный Qt-бэкенд).

Или установка из requirements.txt:

```bash
pip install -r requirements.txt
```

## Запуск

```bash
python butterfly_mgl.py
```

## Управление

| Клавиша | Действие |
|---------|----------|
| **W** | Вперёд по Z |
| **S** | Назад по Z |
| **A** | Влево (X) |
| **D** | Вправо (X) |
| **Q** | Вниз (Y) |
| **E** | Вверх (Y) |
| **Shift** | Ускорение (буст) |
| **SPACE** | Выбор бабочки (цикл по рою) |
| **TAB** | Подсказка по кнопкам и статистика выбранной бабочки |
| **V** | Режим камеры: FREE → CHASE → FPV |
| **H** | Вкл/выкл HUD (полоска здоровья) |
| **1** / **2** | Уменьшить / увеличить плотность туманности |

## Идея 4D

Координата **W** — четвёртое измерение. Трёхмерная сцена — это проекция 4D-пространства. Формула проекции:
- `(x, y, z, w) → (x, y, z) * d / (d - w)` — перспективная проекция «вглубь» экрана.
- Бабочки имеют 4D-координаты `center4` и скорость `vel4`. Их форма строится в 4D и проецируется в 3D.
- Звёзды пока находятся только в 3D; в будущем возможно расширение до 4D.

## Структура проекта

- `butterfly_mgl.py` — основной скрипт (рекомендуется)
- `requirements.txt` — зависимости
- `old/` — архивные версии


## Public source status — 2026-10-29

See [PUBLICATION.md](PUBLICATION.md) for the current verification scope, prototype limitations and runtime boundaries. Earlier audit checklists describe historical work.
