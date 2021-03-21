# _*_ coding: utf-8 _*_
# @author: anniequ
# @file: test.py
# @time: 2020/12/29 16:43
# @Software: PyCharm

from model import *
from data import *


model = unet(pretrained_weights="unet_weights_01-0.159.hdf5")

testGene = testGenerator("D:\project\X\qjpdata\\test")
results = model.predict_generator(testGene, 30, verbose=1)
saveResult("D:\project\X\qjpdata\\test", results)
