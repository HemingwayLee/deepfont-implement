import string
import random
import glob
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def get_random_text(len):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(len))


def create_image(size, message, font):
    width, height = size
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    w, h = font.getsize(message)
    draw.text(((width-w)/2, (height-h)/2), message, font=font, fill='black')
    return image


def main(count):
    font_files = glob.glob("fonts/*")
    for font_file in font_files:
        foldername = Path(font_file).stem
        Path(f"train_data/{foldername}").mkdir(parents=True, exist_ok=True)

        for c in range(count):
            height = 80
            width = 600
            fontsize = 48
            font = ImageFont.truetype(font_file, fontsize)
            msg = get_random_text(random.randint(6, 18))
            pil_img = create_image((width, height), msg, font)
            pil_img.save(f"train_data/{foldername}/{msg}.jpg")
            # plt.imshow(pil_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Put data creation parameters')
    parser.add_argument('--count','-c',required=True)

    args = parser.parse_args()
    main(int(args.count))

