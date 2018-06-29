import numpy as np
import itertools
import cv2
import pickle
import sys
from sklearn.neural_network import MLPRegressor
import datetime

def symm_check(fname):
    img = cv2.imread(fname)
    r_img = cv2.resize(img,(28,28))
    h = r_img.shape[0]
    w= r_img.shape[1]
    img = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)
    ret,img = cv2.threshold(img,127,255,cv2.THRESH_OTSU)
    for i in range(h):
        for j in range(w):
            a=img[i,j]        
            if a<127:
                img[i,j] = 1
            else:
                img[i,j] = 0
    x = np.array(img)
    y = np.fliplr(x)
    z = np.abs(x-y)
    count = 0
    for i in np.nditer(z):
        if i == 1:
            count += 1
    val1 = count/(28*28)
#HORIZONTAL SYMMETRY
    y = np.flipud(x)
    z = np.abs(x-y)

    count = 0
    for i in np.nditer(z):
        if i == 1:
            count += 1
    val2 = count/(28*28)
    if val1 < 0.07 and val2 < 0.03:
        #total
        return 0
    elif val2 < 0.03:
        #horizaontal
        return 3
    elif val1 < 0.07:
        #vertical
        return 1
    else:
        return 2

for i in range(12):
	fname = "t/" + str(i) + ".png"
	im = cv2.imread(fname,0)
	im2 = cv2.resize(im,(28,28))
	ret,th = cv2.threshold(im2,127,255,cv2.THRESH_BINARY)
	for x in np.nditer(th, op_flags=['readwrite']):
		if(x == 255):
			x[...] = 1
		elif(x == 0):
			x[...] = 0
	th = th.tolist()
	th = list(itertools.chain.from_iterable(th))
	th = np.array(th)
	if i == 0:
		inp = th
	else:
		inp = np.vstack((inp, th))

y = [[1,0,0,0,0,0,0],#0
[0,1,0,0,0,0,0],#1
[0,0,1,0,0,0,0],#2
[0,0,0,1,0,0,0],#3
[0,0,0,0,1,0,0],#4
[0,0,0,0,0,1,0],#5
[1,0,0,0,0,0,0],#6
[0,1,0,0,0,0,0],#7
[0,0,1,0,0,0,0],#8
[0,0,0,1,0,0,0],#9
[0,0,0,0,0,0,1],
[0,0,0,0,0,1,0]
]

tsnn = MLPRegressor(solver='adam',activation='relu',beta_1=0.1,tol = 1e-5, alpha=1e-5, hidden_layer_sizes=(2000, 2500), random_state=1)

y = np.array(y)

s_time = datetime.datetime.now()
tsnn.fit(inp,y)
e_time = datetime.datetime.now()
tsnn_time = e_time - s_time
print("tsnn",tsnn_time)

'''
filehandler = open("tsnn.obj","wb")
pickle.dump(tsnn,filehandler)
filehandler.close()
'''


for i in range(29):
	fname = "v/" + str(i) + ".png"
	im = cv2.imread(fname,0)
	im2 = cv2.resize(im,(28,28))
	ret,th = cv2.threshold(im2,127,255,cv2.THRESH_BINARY)
	for x in np.nditer(th, op_flags=['readwrite']):
		if(x == 255):
			x[...] = 1
		elif(x == 0):
			x[...] = 0
	th = th.tolist()
	th = list(itertools.chain.from_iterable(th))
	th = np.array(th)
	if i == 0:
		inp = th
	else:
		inp = np.vstack((inp, th))

y = [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#0
[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#1
[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#2
[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#3
[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#4
[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],#5
[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],#6
[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],#7
[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],#8
[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],#9
[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],#10
[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],#11
[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],#12
[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],#13
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],#14
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],#15
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],#16
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],#17
[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#18
[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#19
[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#20
[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],#21
[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],#22
[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],#23
[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],#24
[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],#25
[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],#26
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],#27
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],#28
]

vsnn = MLPRegressor(solver='adam',activation='relu',beta_1=0.1,tol = 1e-5, alpha=1e-5, hidden_layer_sizes=(2000, 2500), random_state=1)

y = np.array(y)

s_time = datetime.datetime.now()
vsnn.fit(inp,y)
e_time = datetime.datetime.now()
vsnn_time = e_time - s_time
print("vsnn",vsnn_time)

'''
filehandler = open("vsnn.obj","wb")
pickle.dump(vsnn,filehandler)
filehandler.close()
'''

for i in range(78):
	fname = "n/" + str(i) + ".png"
	im = cv2.imread(fname,0)
	im2 = cv2.resize(im,(28,28))
	ret,th = cv2.threshold(im2,127,255,cv2.THRESH_BINARY)
	for x in np.nditer(th, op_flags=['readwrite']):
		if(x == 255):
			x[...] = 1
		elif(x == 0):
			x[...] = 0
	th = th.tolist()
	th = list(itertools.chain.from_iterable(th))
	th = np.array(th)
	if i == 0:
		inp = th
	else:
		inp = np.vstack((inp, th))

y = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #0
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #1
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #2
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #3
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #4
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #5
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #6
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #7
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #8
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #9
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #10
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #11
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #12
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #13
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #14
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #15
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #16
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #17
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #18
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #19
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #20
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #21 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #22
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #23
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #24
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #25
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #26
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #27
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #28
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #29 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #30
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #31
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,0,0,0,0,0,0,0,0,0,0], #32
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,0,0,0,0,0,0,0,0,0,0], #33
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #34
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,0,0,0,0,0,0,0,0,0,0], #35
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1,0,0,0,0,0,0,0,0,0], #36
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #37
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #38
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #39
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #40
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #41
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,1,0,0,0,0,0,0,0,0], #42
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #43
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #44
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #45
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #46
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,1,0,0,0,0,0,0,0], #47
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,1,0,0,0,0,0,0], #48
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #49
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #50
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #51
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,1,0,0,0,0,0], #52
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #53
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,1,0,0,0,0], #54
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #55
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #56
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #57
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #58
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #59
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #60
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #61
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #62
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #63
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #64
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,1,0,0,0], #65
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,1,0,0], #66
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,1,0], #67
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #68
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #69
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #70
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #71
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #72
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #73
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,0,0,0,0,0,0,0,0,0,0], #74
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,0,0,0,0,0,0,0,0,0,0], #75
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,1], #76
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,0,0,0,0,0,0,0,0,0,0], #77
]

nsnn = MLPRegressor(solver='adam',activation='relu',beta_1=0.1,tol = 1e-5, alpha=1e-5, hidden_layer_sizes=(2000, 2500), random_state=1)

y = np.array(y)

s_time = datetime.datetime.now()
nsnn.fit(inp,y)
e_time = datetime.datetime.now()
nsnn_time = e_time - s_time
print("nsnn",nsnn_time)

'''
filehandler = open("nsnn.obj","wb")
pickle.dump(nsnn,filehandler)
filehandler.close()
'''


for i in range(5):
	fname = "h/" + str(i) + ".png"
	im = cv2.imread(fname,0)
	im2 = cv2.resize(im,(28,28))
	ret,th = cv2.threshold(im2,127,255,cv2.THRESH_BINARY)
	for x in np.nditer(th, op_flags=['readwrite']):
		if(x == 255):
			x[...] = 1
		elif(x == 0):
			x[...] = 0
	th = th.tolist()
	th = list(itertools.chain.from_iterable(th))
	th = np.array(th)
	if i == 0:
		inp = th
	else:
		inp = np.vstack((inp, th))

y = [[1,0,0,0],
[0,1,0,0],
[0,0,1,0],
[0,0,0,1],
[1,0,0,0]
]

hsnn = MLPRegressor(solver='adam',activation='relu',beta_1=0.1,tol = 1e-5, alpha=1e-5, hidden_layer_sizes=(2000, 2500), random_state=1)

y = np.array(y)

s_time = datetime.datetime.now()
hsnn.fit(inp,y)
e_time = datetime.datetime.now()
hsnn_time = e_time - s_time
print("hsnn",hsnn_time)

'''
filehandler = open("hsnn.obj","wb")
pickle.dump(hsnn,filehandler)
filehandler.close()
'''

#end of training










#Beginning of char & line extraction

img = cv2.imread(str(sys.argv[1]))
h = img.shape[0]
w= img.shape[1]
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,img = cv2.threshold(img,127,255,cv2.THRESH_OTSU)

for i in range(h):
    for j in range(w):
        a=img[i,j]        
        if a<127:
            img[i,j] = 255
        else:
            img[i,j] = 0

sum = [0 for x in range(h)]
for i in range(h):
    for j in range(w):
        sum[i]+=img[i,j]
        
end = 0
crops = []

j = 0
while j<h:
    beg=j
    if (sum[j]==0):
        while j<(len(sum)):
            if sum[j]==0:                
                j+=1
            else:
                break
        line = int(beg+(j-beg-1)/2)   
    crops.append(line)
    j+=1


x = np.array(crops)
points = np.unique(x)


names = 'crop'
sep = '-'
ext = '.png'
for i in range(len(points)-1):
    num = str(i)
    crop = img[points[i]:points[i+1] , 0:w]
    cv2.imwrite(names+num+ext,crop)


#
# character extraction
#


nol = len(points) - 1
noc = [0 for x in range(nol)]
ccount = 0
spaces = [[] for x in range(nol)]
for x in range(nol):
    crops = []    
    sum = [0 for x in range(w)]
    num = str(x)
    img = cv2.imread(names+num+ext)    
    h = img.shape[0]
    w= img.shape[1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(w):
        for j in range(h):
            sum[i]+=img[j,i]

    i=0
    while sum[i]==0:
        i+=1
    crops.append(i-1)
    beg = i-1
    
    while i<len(sum):
        if sum[i]==0:            
            crops.append(i)
        i+=1
    
    i=0
    while i<(len(crops)-1):
        if (crops[i+1]-crops[i])<=1:
            crops.remove(crops[i])
            i-=1
        i+=1
    crops.remove(crops[i])
    i = w-1
    while sum[i]==0:
        i-=1
    crops.append(i+2)
    nd = i+2
    s_c = []
    i = beg
    while i<nd-1:
        sc = 0
        a = i
        while sum[a]==0:
            sc += 1
            a += 1
        if sc >=18:
            #print(i)
            s_c.append(i)
            i += sc
        i += 1
    
    nam = 'char'
    ex = '.png'
    j = 0
    for i in range(len(crops)-1):     
        n = str(i)
        #print(crops[i])        
        crop = img[0:h , crops[i]:crops[i+1]]
        cv2.imwrite(nam+num+sep+n+ex, crop)
        noc[ccount] += 1
        if j<=len(s_c)-1:
            #print(ccount)
            if s_c[j]>=crops[i] and s_c[j]<crops[i+1]:                
                s_c[j] = noc[ccount]
                j += 1
                #print(crops[i])
                
    spaces[x] = s_c
    ccount +=1

print(spaces)


############################
#      optimize character
############################

names = 'char'

for x in range(nol):
    for y in range(noc[x]):
        crops = []
        num = str(x)
        n = str(y)
        img = cv2.imread(names+num+sep+n+ext)    
        h = img.shape[0]
        w= img.shape[1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sum = [0 for x in range(w)]
        for i in range(w):
            for j in range(h):
                sum[i]+=img[j,i]
        i = 0
        while sum[i]==0:
            i+=1
        crops.append(i-1)
        
        i = w-1
        while sum[i]==0:
            i-=1
        crops.append(i+2)

        for i in range(len(crops)-1):     
            crop = img[0:h , crops[i]:crops[i+1]]
            cv2.imwrite(names+num+sep+n+ext, crop)
            
        crops = []
        img = cv2.imread(names+num+sep+n+ext)    
        h = img.shape[0]
        w= img.shape[1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sum = [0 for x in range(h)]
        for i in range(h):
            for j in range(w):
                sum[i]+=img[i,j]
        i = 0
        while sum[i]==0:
            i+=1
        crops.append(i-1)
        
        i = h-1
        while sum[i]==0:
            i-=1
        crops.append(i+2)

        for i in range(len(crops)-1):     
            crop = img[crops[i]:crops[i+1] , 0:w]
            cv2.imwrite(names+num+sep+n+ext, crop)

#end of char & line extraction 












#beginning of output calculation and output to file
#nol number of lines noc is array

print(nol)
print(noc)
f =  open("output_final.txt","w")
for i in range(nol):
	spaces_iterator = 0
	spaces_interval_count = 0
	for j in range(noc[i]):
		indexing_list = []
		sortable_list = []
		fname = "char" + str(i) + "-" + str(j) + ".png"
		print(fname)
		symm_n = symm_check(fname)
		print(symm_n)
		aaa = cv2.imread(fname,0)
		aaa2 = cv2.resize(aaa,(28,28))
		ret,ath = cv2.threshold(aaa2,127,255,cv2.THRESH_BINARY)
		for x in np.nditer(ath, op_flags=['readwrite']):
			if(x == 255):
				x[...] = 1
			elif(x == 0):
				x[...] = 0
		ath = ath.flatten()
		a22 = ath
		#adding the rest of the sparse matrix
		zzz = np.zeros([1,784])
		zzz = zzz.tolist()
		zzz = list(itertools.chain.from_iterable(zzz))
		zzz = np.array(zzz)
		#for total symmetry
		if symm_n == 0:
			for ii in range(11):
				ts_ath = np.vstack((ath, zzz))
			ou = np.array(tsnn.predict(ts_ath))
			max1 = 0
			mv1 = 0
			for v1 in range(7):
				if ou[0][v1] > max1:
					mv1 = v1
					max1 = ou[0][v1]
			if mv1 == 0:
				tsnn_char = "H"
			elif mv1 == 1:
				tsnn_char = "I"
			elif mv1 == 2:
				tsnn_char = "O"
			elif mv1 == 3:
				tsnn_char = "l"
			elif mv1 == 4:
				tsnn_char = "o"
			elif mv1 == 5:
				tsnn_char = "0"
			elif mv1 == 6:
				tsnn_char = "1"
			f =  open("output_final.txt","a")
			print(tsnn_char)
			f.write(tsnn_char)
		#for vertical symmetry
		elif symm_n == 1:
			for ii in range(18):
				vs_ath = np.vstack((ath, zzz))
			ou = np.array(vsnn.predict(vs_ath))
			max1 = 0
			mv1 = 0
			for v1 in range(19):
				if ou[0][v1] > max1:
					mv1 = v1
					max1 = ou[0][v1]
			if mv1 == 0:
				vsnn_char = "A"
			elif mv1 == 1:
				vsnn_char = "M"
			elif mv1 == 2:
				vsnn_char = "Q"
			elif mv1 == 3:
				vsnn_char = "T"
			elif mv1 == 4:
				vsnn_char = "U"
			elif mv1 == 5:
				vsnn_char = "V"
			elif mv1 == 6:
				vsnn_char = "W"
			elif mv1 == 7:
				vsnn_char = "X"
			elif mv1 == 8:
				vsnn_char = "Y"
			elif mv1 == 9:
				vsnn_char = "e"
			elif mv1 == 10:
				vsnn_char = "i"
			elif mv1 == 11:
				vsnn_char = "m"
			elif mv1 == 12:
				vsnn_char = "n"
			elif mv1 == 13:
				vsnn_char = "u"
			elif mv1 == 14:
				vsnn_char = "v"
			elif mv1 == 15:
				vsnn_char = "w"
			elif mv1 == 16:
				vsnn_char = "x"
			elif mv1 == 17:
				vsnn_char = "8"
			elif mv1 == 18:
				vsnn_char = "o"
			f =  open("output_final.txt","a")
			print(vsnn_char)
			f.write(vsnn_char)
		#for no symmetry
		elif symm_n == 2:
			for ii in range(77):
				ns_ath = np.vstack((ath, zzz))
			ou = np.array(nsnn.predict(ns_ath))
			max1 = 0
			mv1 = 0
			for v1 in range(45):
				if ou[0][v1] > max1:
					mv1 = v1
					max1 = ou[0][v1]
			if mv1 == 0:
			    nsnn_char = "g"
			elif mv1 == 1:
			    nsnn_char = "h"
			elif mv1 == 2:
			    nsnn_char = "S"
			elif mv1 == 3:
			    nsnn_char = "a"
			elif mv1 == 4:
			    nsnn_char = "y"
			elif mv1 == 5:
			    nsnn_char = "B"
			elif mv1 == 6:
			    nsnn_char = "F"
			elif mv1 == 7:
			    nsnn_char = "G"
			elif mv1 == 8:
			    nsnn_char = "J"
			elif mv1 == 9:
			    nsnn_char = "K"
			elif mv1 == 10:
			    nsnn_char = "L"
			elif mv1 == 11:
			    nsnn_char = "N"
			elif mv1 == 12:
			    nsnn_char = "P"
			elif mv1 == 13:
			    nsnn_char = "R"
			elif mv1 == 14:
			    nsnn_char = "Z"
			elif mv1 == 15:
			    nsnn_char = "b"
			elif mv1 == 16:
			    nsnn_char = "d"
			elif mv1 == 17:
			    nsnn_char = "f"
			elif mv1 == 18:
			    nsnn_char = "j"
			elif mv1 == 19:
			    nsnn_char = "k"
			elif mv1 == 20:
			    nsnn_char = "p"
			elif mv1 == 21:
			    nsnn_char = "q"
			elif mv1 == 22:
			    nsnn_char = "r"
			elif mv1 == 23:
			    nsnn_char = "s"
			elif mv1 == 24:
			    nsnn_char = "t"
			elif mv1 == 25:
			    nsnn_char = "z"
			elif mv1 == 26:
			    nsnn_char = "1"
			elif mv1 == 27:
			    nsnn_char = "2"
			elif mv1 == 28:
			    nsnn_char = "3"
			elif mv1 == 29:
			    nsnn_char = "4"
			elif mv1 == 30:
			    nsnn_char = "5"
			elif mv1 == 31:
			    nsnn_char = "6"
			elif mv1 == 32:
			    nsnn_char = "7"
			elif mv1 == 33:
			    nsnn_char = "9"
			elif mv1 == 34:
			    nsnn_char = "C"
			elif mv1 == 35:
			    nsnn_char = "E"
			elif mv1 == 36:
			    nsnn_char = "M"
			elif mv1 == 37:
			    nsnn_char = "W"
			elif mv1 == 38:
			    nsnn_char = "X"
			elif mv1 == 39:
			    nsnn_char = "c"
			elif mv1 == 40:
			    nsnn_char = "e"
			elif mv1 == 41:
			    nsnn_char = "u"
			elif mv1 == 42:
			    nsnn_char = "w"
			elif mv1 == 43:
			    nsnn_char = "x"
			elif mv1 == 44:
			    nsnn_char = "8"
			f =  open("output_final.txt","a")
			print(nsnn_char)
			f.write(nsnn_char)
		#for horizontal symmetry
		elif symm_n == 3:
			for ii in range(4):
				hs_ath = np.vstack((ath, zzz))
			ou = np.array(hsnn.predict(hs_ath))
			max1 = 0
			mv1 = 0
			for v1 in range(4):
				if ou[0][v1] > max1:
					mv1 = v1
					max1 = ou[0][v1]
			if mv1 == 0:
				hsnn_char = "D"
			elif mv1 == 1:
				hsnn_char = "E"
			elif mv1 == 2:
				hsnn_char = "c"
			elif mv1 == 3:
				hsnn_char = "C"
			elif mv1 == 0:
				hsnn_char = "D"
			f =  open("output_final.txt","a")
			print(hsnn_char)
			f.write(hsnn_char)
		if len(spaces[i]) != 0:
			spaces_interval_count = spaces_interval_count + 1
			if spaces_interval_count == spaces[i][spaces_iterator]:
				f =  open("output_final.txt","a")
				f.write(" ")
				if spaces_iterator < (len(spaces[i])-1):
					spaces_iterator = spaces_iterator + 1
		#sortable_list = indexing_list
		#sortable_list.sort()
		#print(indexing_list)
		#print(sortable_list)
		#chosen_index = 3
		#print(indexing_list)
		#print(sortable_list)
		#max_nn_activation = indexing_list.index(sortable_list[chosen_index])
		#if max_nn_activation == 0:
		#elif max_nn_activation == 1:
		#	f.write(vsnn_char)
		#elif max_nn_activation == 2:
		#	f.write(nsnn_char)
		#elif max_nn_activation == 3:
		#	f.write(hsnn_char)
	f.write("\n")
f.close()













































