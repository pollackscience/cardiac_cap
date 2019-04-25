import os
import sys
import glob
import shutil


def make_subdirs(input_dir=None):
    if not input_dir:
        input_dir = 'data/DET0000201/SA'
    file_list = glob.glob(input_dir+'/*.dcm')
    file_png = glob.glob(input_dir+'/*.png')
    unique_SAs = set([a.split('_')[1] for a in file_list]+[a.split('_')[1] for a in file_png])
    print(unique_SAs)

    for SA in unique_SAs:
        try:
            os.mkdir(input_dir+'/'+SA)
        except:
           pass
    for f in file_list:
        SA = f.split('_')[1]
        try:
            shutil.move(f, input_dir+'/'+SA+'/'+f.split('/')[-1])
        except:
            pass
    for f in file_png:
        SA = f.split('_')[1]
        try:
            shutil.move(f, input_dir+'/'+SA+'/'+f.split('/')[-1])
        except:
            pass

if __name__ == '__main__':
    make_subdirs(sys.argv[1])
