from model import *
from data import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')
myGene = trainGenerator(2, 'data/train', 'image', 'label', data_gen_args, save_to_dir=None)


model = unet()
model_checkpoint = ModelCheckpoint('unet_weights_{epoch:02d}-{loss:.3f}.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=200, epochs=1, callbacks=[model_checkpoint])

testGene = testGenerator("data/test")
results = model.predict_generator(testGene, 30, verbose=1)
saveResult("data/test", results)

"""
keras.callbacks.ModelCheckpoint() parameters:
ModelCheckpoint(filepath,                   # 字符串，保存模型的路径，filepath可以是格式化的字符串
				monitor='val_loss',         # 需要监视的值，通常为：val_acc 或 val_loss 或 acc 或 loss
				verbose=0,                  # 信息展示模式，0或1。为1表示输出epoch模型保存信息，默认为0表示不输出该信息
				save_best_only=False,       # 当设置为True时，将只保存在验证集上性能最好的模型
				save_weights_only=False,    # 若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
				mode='auto',                # ‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则
				period=1)                   # CheckPoint之间的间隔的epoch数

fit_generator(self, 
              generator,                # 生成器函数，或者一个Sequence(keras.utils.Sequence)对象的实例。生成器的输出应该为: (inputs, targets)的tuple；
              steps_per_epoch,          # int，当生成器返回steps_per_epoch次数据时一个epoch结束，执行下一个epoch， int(number_of_train_samples / batch_size)；
              epochs=1,                 # int，训练的epoch数             
              verbose=1,                # 日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
              callbacks=None,           # 
              validation_data=None,     # 验证集，与generator类似
              validation_steps=None,    #
              class_weight=None,        # 规定类别权重的字典，将类别映射为权重，常用于处理样本不均衡问题。
              max_q_size=10,            # 生成器队列的最大容量
              workers=1,                # 最大进程数
              pickle_safe=False,        #
              use_multiprocessing=False,# 是否基于进程的多线程。
              shuffle=True,             # 是否在每轮迭代之前打乱 batch 的顺序。 只能与Sequence(keras.utils.Sequence) 实例同用。
              initial_epoch=0)          # 从该参数指定的epoch开始训练，在继续之前的训练时有用。
"""
