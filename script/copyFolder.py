import glob as glob
import shutil
error_ids = {'281', '571', '2064', '2408', '2247', '2310', '6023', '6328', '2086', '1022', '626', '8460', '1110', '748', '10249', '2346', '1000', '6036', '12424', '1300', '12490'}

count = 0
for error_id in error_ids:
    path = '/media/wall/4TB_HDD/full_dataset/0511_dataset/pill0603/train_aug/{pill_id}/{pill_id}_*.png'.format(pill_id=error_id)
    for file in glob.glob(path):
        name = file.split('/')[-1]
        dst = '../pill_dataset/train/{pill_id}/'.format(pill_id=error_id) + name
        print(dst)
        shutil.copyfile(file, dst)








