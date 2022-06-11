import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = THIS_FOLDER+'/img/0'

file_names = os.listdir(IMAGE_FOLDER)

for name in file_names:
    src = os.path.join(IMAGE_FOLDER, name)
    pos = name.find("1")
    dst = "간판_가로형간판"+name[pos-1:]
    dst = os.path.join(IMAGE_FOLDER, dst)
    os.rename(src, dst)