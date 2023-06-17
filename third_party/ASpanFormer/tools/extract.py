import os
import glob
from re import split
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

scannet_dir='/root/data/ScanNet-v2-1.0.0/data/raw'
dump_dir='/root/data/scannet_dump'
num_process=32

def extract(seq,scannet_dir,split,dump_dir):
    assert split=='train' or split=='test'
    if not os.path.exists(os.path.join(dump_dir,split,seq)):
            os.mkdir(os.path.join(dump_dir,split,seq))
    cmd='python reader.py --filename '+os.path.join(scannet_dir,'scans' if split=='train' else 'scans_test',seq,seq+'.sens')+' --output_path '+os.path.join(dump_dir,split,seq)+\
            ' --export_depth_images --export_color_images --export_poses --export_intrinsics'
    os.system(cmd)

if __name__=='__main__':
    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)
        os.mkdir(os.path.join(dump_dir,'train'))
        os.mkdir(os.path.join(dump_dir,'test'))

    train_seq_list=[seq.split('/')[-1] for seq in glob.glob(os.path.join(scannet_dir,'scans','scene*'))]
    test_seq_list=[seq.split('/')[-1] for seq in glob.glob(os.path.join(scannet_dir,'scans_test','scene*'))]

    extract_train=partial(extract,scannet_dir=scannet_dir,split='train',dump_dir=dump_dir)
    extract_test=partial(extract,scannet_dir=scannet_dir,split='test',dump_dir=dump_dir)

    num_train_iter=len(train_seq_list)//num_process if len(train_seq_list)%num_process==0 else len(train_seq_list)//num_process+1
    num_test_iter=len(test_seq_list)//num_process if len(test_seq_list)%num_process==0 else len(test_seq_list)//num_process+1

    pool = Pool(num_process)
    for index in tqdm(range(num_train_iter)):
        seq_list=train_seq_list[index*num_process:min((index+1)*num_process,len(train_seq_list))]
        pool.map(extract_train,seq_list)
    pool.close()
    pool.join()
    
    pool = Pool(num_process)
    for index in tqdm(range(num_test_iter)):
        seq_list=test_seq_list[index*num_process:min((index+1)*num_process,len(test_seq_list))]
        pool.map(extract_test,seq_list)
    pool.close()
    pool.join()