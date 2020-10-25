# ミニマムニューロン #

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import InputLayer
from keras import optimizers
import numpy

model = Sequential()
model.add( InputLayer( input_shape= (1,) ) )
model.add( Dense( units= 1, activation='linear') ) # units(層内のニューロンの数) activation(活性化関数)
# 重みを探す方法（＝最適化法）を指定 optimizersはモジュール
sgd = optimizers.SGD( learning_rate=0.001 ) # learning_rate(学習率)の設定は大切！
# sgd = optimizers.SGD( learning_rate=0.000000000000000001 ) # learning_rate(学習率)の設定は大切！
model.compile( loss = 'mse',optimizer = sgd, metrics =["accuracy"]) # 二乗平均誤差

#概略表示
model.summary()

#訓練開始
train_x0 = numpy.array( [ 10, 14, 8, 16 ] )
train_x1 = numpy.array( [ 5, 7, 4, 8 ] )
history = model.fit( train_x0, train_x1, epochs = 50 )

#実用テスト
num1 = input()
num2 = input()
# test_x0 = numpy.array( [ 100, 200 ] )
test_x0 = numpy.array( [ int(num1) ,int(num2) ] )
res = model.predict( test_x0 )
print( res )