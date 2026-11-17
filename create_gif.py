import argparse
import os
from PIL import Image


def create_gif(folder_path, duration, output_name='output.gif'):
  # Поддерживаемые форматы изображений
  valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

  # Собираем и сортируем файлы из указанной папки
  try:
    files = sorted([
        f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)
    ])
  except FileNotFoundError:
    print(f"Ошибка: Папка '{folder_path}' не найдена.")
    return

  if not files:
    print(f"В папке '{folder_path}' не найдено подходящих изображений.")
    return

  # Открываем изображения
  images = []
  for file in files:
    img_path = os.path.join(folder_path, file)
    images.append(Image.open(img_path))

  # Сохраняем результат
  images[0].save(
      output_name,
      format='GIF',
      save_all=True,
      append_images=images[1:],
      duration=duration,
      loop=0,
      optimize=True,
  )
  print(f'Готово! GIF сохранен как {output_name} (кадров: {len(images)})')


if __name__ == '__main__':
  # Настраиваем прием аргументов из командной строки
  parser = argparse.ArgumentParser(description='Создание GIF из папки с фото.')
  parser.add_argument('--image_dir', type=str, help='Путь к папке с изображениями')
  parser.add_argument(
      '--duration', type=int, help='Длительность одного кадра в миллисекундах'
  )
  parser.add_argument(
      '-o',
      '--output',
      type=str,
      default='output.gif',
      help='Имя выходного файла (по умолчанию output.gif)',
  )

  args = parser.parse_args()
  create_gif(args.image_dir, args.duration, args.output)
