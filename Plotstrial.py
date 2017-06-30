#from sklearn.utils import shuffle
#from IPython.display import Image
from IPython.display import display
import cv2
import tensorflow as tf
import time
import matplotlib.pyplot as plt

#plt.figure()
EPOCHS=[1,2,3,4,5,6,7,8]
pts_train_acc=[3,10,80,90,91,92,91,92]
pts_valid_acc=[2,15,85,86,90,94,95,94]
pts_train_loss=[5,25,75,82,84,88,89,91]
pts_valid_loss=[1,20,71,77,81,84,85,95]

train_acc_plt,=plt.plot(EPOCHS, pts_train_acc, 'r', label='Training Accuracy')
valid_acc_plt,=plt.plot(EPOCHS, pts_valid_acc, 'b', label='Validation Accuracy')
#plt.plot(EPOCHS, pts_train_acc, 'r', label='Training Accuracy')
#plt.plot(EPOCHS, pts_valid_acc, 'b', label='Validation Accuracy')
plt.xlabel('EPOCHS')
plt.ylabel('ACCURACY')
plt.title('ACCURACY vs. EPOCH')
plt.legend([train_acc_plt, valid_acc_plt], loc='best')
#plt.legend(loc='best')
plt.show()

#train_loss_plt,=plt.plot(EPOCHS, pts_train_loss, 'r', label='Training Loss')
#valid_loss_plt,=plt.plot(EPOCHS, pts_valid_loss, 'b', label='Validation Loss')
plt.plot(EPOCHS, pts_train_loss, 'r', label='Training Loss')
plt.plot(EPOCHS, pts_valid_loss, 'b', label='Validation Loss')
plt.xlabel('EPOCHS')
plt.ylabel('LOSS')
plt.title('ACCURACY vs. LOSS')
#plt.legend([train_loss_plt, valid_loss_plt], loc='best')
plt.legend(loc='best')
plt.show()