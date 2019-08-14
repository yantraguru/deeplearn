import io
import os
import glob
from pathlib import Path
from PIL import Image

src_dir = './../data/caps_and_shoes_extended/'
dest_dir = './../data/caps_and_shoes_squared/'

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

error_count = 0
for filetype in ['**/*.jpg','**/*.jpeg']:
    for filename in glob.iglob(src_dir + '**/*.*', recursive=True):
        try:
            im = Image.open(filename)
        except OSError as oe:
            print('Error opening image : %s'% filename)
            with open(filename, "rb") as f:
                try:
                    b = io.BytesIO(f.read())
                    im = Image.open(b)
                except:
                    error_count+=1
                    print('Error count : %d // could not read %s'%(error_count,filename))
                    continue 
 	
        if im.mode in ('RGBA'):
            im = im.convert("RGB")

        im_crop = crop_max_square(im)
        dest_path = filename.replace(src_dir,dest_dir)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        im_crop.save(dest_path, quality=95)
