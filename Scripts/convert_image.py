#Convert a .jpg file to image buffer. 
import os
import glob
import numpy as np
from skimage import color
from skimage import io
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default=\
        "C:/Users/User/Desktop/LeNet_5/Test_Dataset",
        help='path to image dir')
parser.add_argument('--out_dir', type=str, default=\
        ".",
        help='path to output dir')

cmd_args, _ = parser.parse_known_args()
image_dir = cmd_args.image_dir
out_dir = cmd_args.out_dir

input_dir = out_dir+'/Input'
log_dir = out_dir+'/Log'

if (not os.path.isdir(out_dir)):
    os.makedirs(out_dir)
if (not os.path.isdir(input_dir)):
    os.makedirs(input_dir)
if (not os.path.isdir(log_dir)):
    os.makedirs(log_dir)

max_pic = 80
include_list = out_dir+'/include_list.h'

if (os.path.isfile(include_list)):
    os.remove(include_list)

h=open(include_list,'a')  
for i in range (10):
    j=0
    k=0
    l=0
    tmp = ''
    log = ''

    paths = glob.glob(image_dir + '/'+str(i)+'/*.jpg')
    for image_path in paths:
        j=j+1
        k=k+1

        if (k==1):
            l=l+1
            i_file = 'input_'+str(i)+'_'+str(l)+'.h'
            image_file = input_dir + '/' + i_file
            if (os.path.isfile(image_file)):
                os.remove(image_file)
            f=open(image_file,'a')  
            
            l_file = 'log_'+str(i)+'_'+str(l)+'.h'
            log_file = log_dir + '/' + l_file
            if (os.path.isfile(log_file)):
                os.remove(log_file)
            g=open(log_file,'a')
            if (i==0 and l==1):
                h.write('#include "' + i_file + '"\n')
            else:
                h.write('//#include "' + i_file + '"\n')

            # Reading an image in default mode 
        img = io.imread(image_path)
        image = color.rgb2gray(img)
        npimage = np.round(np.asarray(image))
        npimage = npimage/2
        npimage = npimage.ravel()
        npimage = npimage.astype(int)
        image_list = npimage.tolist()
        list_string = map(str, image_list) 

        string = ','.join(list_string)
        picture = 'DIGIT_IMG_DATA_'+str(j)
        string = '#define ' + picture +' { ' + string + ' }\n\n'
        f.write(string)

        log = picture + ' : ' + image_path +'\n'
        g.write(log)

        if (tmp==''):
            tmp = picture
        else:
            tmp = tmp + ', '+ picture
            
        if (k==max_pic or j==len(paths)):
            string = "#define CLASS " + str(i) + '\n\n'
            string = string + "#define PIC_NUM " + str(k) + '\n\n'
            string = string + "q7_t input_data[PIC_NUM][28*28] = {" + tmp + "};\n\n"
            k=0
            tmp=''
            log=''
            f.write(string)
            f.close()
            g.close()
            
h.close()