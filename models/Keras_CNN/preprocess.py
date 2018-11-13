import cv2
import numpy as np
import sklearn as sk

face_target_size = (48,48)


from os import listdir


posdir = '../../smiles/pos'
negdir = '../../smiles/neg'

pos = listdir(posdir)
neg = listdir(negdir)
print(len(pos),len(neg))

posfaces = np.zeros((len(pos),face_target_size[0]*face_target_size[1]))

Visualise = False
if Visualise:
    print("Press 'q' to stop the iteration!")
    print("Set 'Visualise=False' to speed things up.")

i=0
for p in pos:
    im = cv2.imread(posdir+'/'+p, cv2.IMREAD_GRAYSCALE)

    if Visualise:
        print(p, im.shape)
        cv2.imshow("face",im)
    posfaces[i,:] = im.flatten()
    i += 1
    
    if Visualise:
        k = cv2.waitKey(500) 
        if k!=-1:
            print(k)
        if k & 0xFF == ord('q'):
            break
if Visualise:    
    cv2.destroyWindow("face")

negfaces = np.zeros((len(neg),face_target_size[0]*face_target_size[1]))

i=0
for p in neg:
    im = cv2.imread(negdir+'/'+p, cv2.IMREAD_GRAYSCALE)
    negfaces[i,:] = im.flatten()
    i+=1

print(posfaces.shape,negfaces.shape)

X = np.concatenate((posfaces, negfaces), axis=0)
y = np.concatenate((np.ones((posfaces.shape[0],1)), np.zeros((negfaces.shape[0],1))), axis=0)


permu = np.random.permutation(X.shape[0])
X = X[permu, :]
y = y[permu, :]

X = X/255

np.savetxt("image.csv", X, delimiter=',')
np.savetxt("label.csv", y, delimiter=',')
