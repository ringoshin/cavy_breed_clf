"""
Created on Sun 8 Aug 2019

@author: ringoshin

One-time removal of underscores from some of the image extensions
"""

import os

image_path = 'images'
image_folders = os.listdir(image_path)

print(" > Renaming image file extensions...")
for each_folder in image_folders:
    if not os.path.isdir(os.path.join(image_path,each_folder)):
        continue
        
    image_list_path = os.path.join(image_path, each_folder)
    for each_image in os.listdir(image_list_path):
        fname, fext = os.path.splitext(image_list_path+'/'+each_image)
        old_name = os.path.join(image_list_path, each_image)
        new_name = fname + fext.strip('_')
        
        if old_name!=new_name:
            #print(old_name, new_name)
            os.rename(old_name, new_name)

print(" > done!")