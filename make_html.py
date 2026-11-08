import os
from pathlib import Path

# Папка, где лежат копии файлов
compare_dir = Path.home() / "benchmarking" / "PPT" / "compare"
html_path = compare_dir / "compare.html"
print(compare_dir)
# Собираем список всех PNG
images = sorted(compare_dir.glob("12M2102884_10_7_Fake_B_*.png"))
print(len(images))
# Генерируем HTML
html = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Compare Epochs</title>
<style>
  body { font-family: sans-serif; background: #fafafa; padding: 20px; }
  .img-row { display: flex; align-items: center; margin-bottom: 16px; }
  .img-row img { height: 150px; border-radius: 8px; box-shadow: 0 0 5px rgba(0,0,0,0.3); }
  .epoch { margin-left: 20px; font-size: 18px; font-weight: bold; color: #333; }
</style>
</head>
<body>
<h2>Сравнение изображений по эпохам</h2>
"""

for img in images:
    print(img)
    # Извлекаем название эпохи (часть после _Fake_B_)
    epoch = img.stem.split("_Fake_B_")[-1]
    html += f"""
    <div class="img-row">
        <img src="{img.name}" alt="{epoch}">
        <div class="epoch">{epoch}</div>
    </div>
    """

html += """
<h2> End </h2>
</body>
</html>
"""

# Сохраняем HTML
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ HTML создан: {html_path}")
print("Открой его в браузере, чтобы просмотреть сравнение.")
