from PIL import Image
import random

image_path = './dataset/test/12083/12083_0-0.png'
img = Image.open(image_path)

angle = random.randint(1, 360)
rotate_img = img.rotate(angle)

rotate_img.show()
rotate_img = rotate_img.save("rotate.png")
