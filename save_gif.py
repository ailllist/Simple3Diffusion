import glob
from PIL import Image

imgs = glob.glob("imgs/*.png")
imgs = sorted(imgs, key=lambda img: int(img.split("/")[-1].split(".")[0]), reverse=True)

images = [Image.open(img) for img in imgs]
images[0].save("res.gif", save_all=True, append_images=images[1:], duration=10, loop=0)