# -*- coding: utf-8 -*- 
# @Time : 2024/7/20 16:06 
# @Author : DirtyBoy 
# @File : my_gen_adv.py
import hashlib, subprocess, os
from tqdm import tqdm
import pandas as pd

rmt_dir = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/Lable_noise/MalRadar_variation'
apk_dir = '/home/lhd/ADV_MVML/AVPASS/src/'
DST_DIR = '/mnt/local_sdc1/lhd/MalRadar_variation/'
command = ['python2', 'gen_disguise.py', '-i', '', 'individual', '-o', '', '-v', '']


def calculate_hashes(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def process_item(item, apk_dir, dst_dir, command, v):
    command[3] = os.path.join(apk_dir, item)
    command[6] = os.path.join(dst_dir, item)
    command[-1] = v
    result = subprocess.run(command, capture_output=True, text=True)
    return result


if __name__ == '__main__':
    dst_dir = DST_DIR
    variation_list = os.listdir(dst_dir)
    for v_ in range(19,35):
        malware = os.listdir(apk_dir)
        malware = [item for item in malware if os.path.splitext(item)[1] == '.apk']
        # malware_ = [item.split('.')[0] for item in malware]
        v = 'v' + str(v_ + 1)
        if v in variation_list and 'v' + str(v_ + 2) not in variation_list:
            exist_malware = os.listdir(os.path.join(rmt_dir, v))
            index = 0
            for j, item in enumerate(malware):
                if item in exist_malware:
                    index = j
            print(v)
            for item in tqdm(malware[index + 1:], desc="Generating malware variation"):
                command[3] = os.path.join(apk_dir, item)
                command[6] = os.path.join(os.path.join(dst_dir, v), item)
                command[-1] = v
                result = subprocess.run(command, capture_output=True, text=True)
        elif v not in variation_list and 'v' + str(v_) in variation_list:
            malware = [item for item in malware if not os.path.isfile(os.path.join(os.path.join(dst_dir, v), item))]
            print(v)
            for item in tqdm(malware, desc="Generating malware variation"):
                command[3] = os.path.join(apk_dir, item)
                command[6] = os.path.join(os.path.join(dst_dir, v), item)
                command[-1] = v
                result = subprocess.run(command, capture_output=True, text=True)
