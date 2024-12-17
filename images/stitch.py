from PIL import Image
import os

def stitch_images_vertically(image_paths, output_path):
    """
    Сшивает изображения вертикально и сохраняет результат.
    
    Args:
        image_paths: Список путей к изображениям.
        output_path: Путь для сохранения итогового изображения.
    """
    # Загружаем изображения
    images = [Image.open(img_path).convert("RGBA") for img_path in image_paths]
    
    # Рассчитываем общую ширину и высоту
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    
    # Создаем итоговое изображение
    stitched_image = Image.new("RGBA", (total_width, max_height), (0, 0, 0, 0))  # Прозрачный фон
    
    # Вставляем изображения одно за другим
    x_offset = 0
    for img in images:
        stitched_image.paste(img, (x_offset, 0), img)
        x_offset += img.width
    
    # Сохраняем итоговое изображение
    stitched_image.save(output_path)
    print(f"Stitched image saved at: {output_path}")

# Пример использования
image_folder = "images"
output_file = os.path.join(image_folder, "stitched_image.png")

# Пути к изображениям
image_paths = [
    os.path.join(image_folder, "Loss_ODE_of_the_second_order.png"),
    os.path.join(image_folder, "Loss_ODE_of_the_first_order.png"),
    os.path.join(image_folder, "Loss_alt_ODE.png"),
]

# Сшиваем изображения
stitch_images_vertically(image_paths, output_file)
