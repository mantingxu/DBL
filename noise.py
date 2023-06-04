import glob
import random
import cv2


def add_noise(img_path):
    img = cv2.imread(img_path)
    # Getting the dimensions of the image
    row, col, ch = img.shape

    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(100, 300)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(100, 300)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord] = 0
    return img


from PIL import Image
import random


def rotateImg(image_path, dst_path):
    img = Image.open(image_path)

    angle = random.randint(1, 360)
    rotate_img = img.rotate(angle)
    rotate_img = rotate_img.save(dst_path)

    return dst_path


# salt-and-pepper noise can
# be applied only to grayscale images
# Reading the color image in grayscale image
# image_path = './dataset/test/12083/12083_0-0.png'
pillId = ['12448', '12222', '12083', '325', '2311', '2321', '4061', '4115', '6356']
for pill in pillId:
    count = 0
    folderPath = './dataset/test/{pillId}/*.png'.format(pillId=pill)
    for file in glob.glob(folderPath):
        img = cv2.imread(file)
        name = file.split('/')[-1].replace('.png', '')
        original = name.split('_')[0]
        save_path = './dataset/aug/{original}_{count}.png'.format(original=original, count=count)
        cv2.imwrite(save_path, img)  # 0

        for i in range(5):
            count += 1
            save_path = './dataset/aug_test/{original}_{count}.png'.format(original=original, count=count)
            dst = rotateImg(file, save_path)
            # count += 1
            # save_path = './dataset/aug/{original}_{count}.png'.format(original=original, count=count)
            cv2.imwrite(save_path, add_noise(dst))  # 1
