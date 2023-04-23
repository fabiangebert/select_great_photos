import os

from PIL import Image, ImageOps


def prepare_image(file, target_width, target_height):
    img = Image.open(file)

    img = ImageOps.exif_transpose(img)

    img = img.convert('RGB')

    # Calculate the new width and height for the right aspect ratio
    width, height = img.size
    current_ratio = float(width) / float(height)
    target_ratio = float(target_width) / float(target_height)
    if current_ratio != target_ratio:
        if current_ratio > target_ratio:
            # Crop the sides
            new_width = int(height * target_ratio)
            left = int((width - new_width) / 2)
            right = int((width + new_width) / 2)
            top, bottom = 0, height
        else:  # portrait orientation
            # Crop the top and bottom
            new_height = int(width / target_ratio)
            top = int((height - new_height) / 2)
            bottom = int((height + new_height) / 2)
            left, right = 0, width

        # Crop the image
        img = img.crop((left, top, right, bottom))

    # Scale using "NEAREST" sampling to keep blur in place
    img.thumbnail((target_width, target_height), Image.Resampling.NEAREST)
    return img


def save_image_in_class_subfolder(img, source_file, target_dir):
    category = os.path.basename(os.path.dirname(source_file))
    target_class_dir = os.path.join(target_dir, category)
    os.makedirs(target_class_dir, exist_ok=True)

    img.save(os.path.join(target_class_dir, os.path.basename(source_file)))


def prepare_image_file(file, target_width, target_height, target_dir):
    img = prepare_image(file, target_width, target_height)
    save_image_in_class_subfolder(img, file, target_dir)
    img.close()
