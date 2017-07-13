from PIL import Image
from glob import glob

path = "../PlayersBoundingBox/Test_Images/"
search_pattern = "/*.png"

im_paths = glob(path+search_pattern)

for p in im_paths:
    im = Image.open(p)
    rgb = im.convert('RGB')
    new_p = p.replace(".png", ".jpg")
    rgb.save(new_p)
print("finished")
