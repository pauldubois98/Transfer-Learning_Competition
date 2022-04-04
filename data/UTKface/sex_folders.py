import os
import shutil

in_folder_name = 'cropped'
out_folder_name = 'sex_subfolders'
img_list = os.listdir(in_folder_name)

i = 0
for img_filename in img_list[:]:
    if 'H' in img_filename:
        shutil.copyfile(in_folder_name+'/'+img_filename,
                        out_folder_name+'/H/'+img_filename)
    if 'F' in img_filename:
        shutil.copyfile(in_folder_name+'/'+img_filename,
                        out_folder_name+'/F/'+img_filename)
    
