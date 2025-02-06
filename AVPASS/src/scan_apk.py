# -*- coding: utf-8 -*- 
# @Time : 2024/7/20 21:46 
# @Author : DirtyBoy 
# @File : scan_apk.py
import os
from tqdm import tqdm
import subprocess


apk_dir = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/CICDataset/source_apk/CIC1719_Malware_variation/'
command = [
    'vt',
    '-f',
    '',
    '-j'
]

if __name__ == '__main__':
    for v_ in range(30,35):
        v = 'v' + str(v_ + 1)
        malware = os.listdir(os.path.join(apk_dir, v))
        for item in tqdm(malware, desc="Scan malware variation"):
            command[2] = os.path.join(os.path.join(apk_dir, v), item)
            result = subprocess.run(command, capture_output=True, text=True)
