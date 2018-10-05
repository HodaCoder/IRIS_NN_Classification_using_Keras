import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.datasets import load_iris
from keras import regularizers
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


save_dir = os.path.join(os.getcwd(), 'saved_models')  # Directory to save the model
model_name = 'Iris_model.h5'                          # Name of the model to be saved
batch_size = 5                                       # Batch size for train
epochs = 100                                         # Number of Epochs for training
RUN_NAME = "Run with " + str(epochs) + " epochs"   # Log file name with different input

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


iris_data_set = load_iris()                     # load the iris dataset

x = iris_data_set.data
y_raw = iris_data_set.target.reshape(-1, 1)        # Convert data to a single column

y = keras.utils.to_categorical(y_raw, iris_data_set.target_names.size)

# Split the data for training and testing
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.20, random_state=13)

# define the model
model = Sequential()
model.add(Dense(20,
                input_shape=x.shape[1:],
                activation='tanh',
                name='Dense_1',
                W_regularizer=regularizers.l2(0.01))  # overfiting mitigation
)
model.add(Dropout(0.25))
model.add(Dense(iris_data_set.target_names.size, activation='softmax', name='output'))


print('Summary of the Neural Network Model : ')
print(model.summary())


# initiate ADAM optimizer
opt = keras.optimizers.Adam(
    lr=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8
)

# Let's compile and train the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

# Create a TensorBoard logger
logger = keras.callbacks.TensorBoard(
    log_dir='logs/{}'.format(RUN_NAME),
    histogram_freq=5,
    write_graph=True
)
keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=0,
    verbose=0,
    mode='auto'
)
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_valid, y_valid),
    shuffle=True,
    callbacks=[logger]
)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model
scores = model.evaluate(x_valid, y_valid, verbose=1)
print('Final Test loss:', scores[0])
print('Final Test accuracy:', scores[1])

# Predicting the validation values
y_pred = model.predict_classes(x_valid, verbose=1)
y_pred_all = model.predict_classes(x, verbose=1)

# wrong prediction items
validation_vector=np.asarray([sum(e) for e in np.multiply(keras.utils.to_categorical(y_pred, iris_data_set.target_names.size),y_valid)])
wrong_prediction_indexes=np.asarray(np.where(validation_vector == 0))
validation_all_vector=np.asarray([sum(e) for e in np.multiply(keras.utils.to_categorical(y_pred_all, iris_data_set.target_names.size), y)])
wrong_prediction_all_indexes=np.asarray(np.where(validation_all_vector == 0))

# summarize history for accuracy
plt.figure(1)
plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
# plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss

plt.subplot(212)
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

'''
# plotting the result for validation data
labelTups = [(iris_data_set.target_names[0], 0), (iris_data_set.target_names[1], 1), (iris_data_set.target_names[2], 2)]
fig = plt.figure(figsize=(10.5, 8))
ax = fig.add_subplot(111, projection='3d', elev=-150, azim=110)
arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)
# plot ox0y0z0 axes


for item in wrong_prediction_indexes.T:
    # print(item)
    ax.text3D(x_valid[:, 0].mean() + np.argmax(y_valid[item]),
              x_valid[:, 1].mean() + 1.5 - np.argmax(y_valid[item]),
              x_valid[:, 2].mean(), 'Ground Truth: '+iris_data_set.target_names[np.argmax(y_valid[item])],
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    a = Arrow3D([x_valid[:, 0].mean()+ np.argmax(y_valid[item]), x_valid[item, 0].mean()],
                [x_valid[:, 1].mean()+ 1.5 - np.argmax(y_valid[item]) , x_valid[item, 1].mean()],
                [x_valid[:, 2].mean(), x_valid[item, 2].mean()], **arrow_prop_dict)
    ax.add_artist(a)

sc=ax.scatter(x_valid[:, 0], x_valid[:, 1], x_valid[:, 2], c=y_pred,
            edgecolor='k', label=[lt[0] for lt in labelTups])
plt.title('IRIS classification via Neural Network')
ax.set_xlabel(iris_data_set.feature_names[0])
ax.set_ylabel(iris_data_set.feature_names[1])
ax.set_zlabel(iris_data_set.feature_names[2])
colors = [sc.cmap(sc.norm(i)) for i in [0, 1, 2]]
custom_lines = [plt.Line2D([], [], ls="", marker='.',
                mec='k', mfc=c, mew=.1, ms=20) for c in colors]
ax.legend(custom_lines, [lt[0] for lt in labelTups],
          bbox_to_anchor=(1.0, .5))
plt.show()
'''

# plotting the result for all data

labelTups = [(iris_data_set.target_names[0], 0), (iris_data_set.target_names[1], 1), (iris_data_set.target_names[2], 2)]
fig = plt.figure(figsize=(6, 4))
plt.rcParams.update({'font.size': 7})
ax = fig.add_subplot(111, projection='3d', elev=-150, azim=110)
arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)
# plot ox0y0z0 axes


for item in wrong_prediction_all_indexes.T:
    # print(item)
    ax.text3D(x[:, 0].mean() + np.argmax(y[item]),
              x[:, 1].mean() + 1.5 - np.argmax(y[item]),
              x[:, 2].mean(), 'Ground Truth: '+iris_data_set.target_names[np.argmax(y[item])],
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    a = Arrow3D([x[:, 0].mean()+ np.argmax(y[item]), x[item, 0].mean()],
                [x[:, 1].mean()+ 1.5 -np.argmax(y[item]) , x[item, 1].mean()],
                [x[:, 2].mean(), x[item, 2].mean()], **arrow_prop_dict)
    ax.add_artist(a)

sc=ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y_pred_all,
            edgecolor='k', label=[lt[0] for lt in labelTups])
plt.title('IRIS classification via Neural Network. The accuracy is: '+ str(1-wrong_prediction_all_indexes.size/validation_all_vector.size))
ax.set_xlabel(iris_data_set.feature_names[0])
ax.set_ylabel(iris_data_set.feature_names[1])
ax.set_zlabel(iris_data_set.feature_names[2])
colors = [sc.cmap(sc.norm(i)) for i in [0, 1, 2]]
custom_lines = [plt.Line2D([], [], ls="", marker='.',
                mec='k', mfc=c, mew=.1, ms=20) for c in colors]
ax.legend(custom_lines, [lt[0] for lt in labelTups],
          bbox_to_anchor=(0.3, .75))
fig.tight_layout()
plt.show()


# comparing with Logistic Regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
lr.fit(x, y_raw)
y_pred_lr = lr.predict(x)

validation_lr_vector=np.asarray([sum(e) for e in np.multiply(keras.utils.to_categorical(y_pred_lr, iris_data_set.target_names.size), y)])
wrong_prediction_lr_indexes=np.asarray(np.where(validation_lr_vector == 0))


# plotting the result for Logistic Regression
labelTups = [(iris_data_set.target_names[0], 0), (iris_data_set.target_names[1], 1), (iris_data_set.target_names[2], 2)]
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111, projection='3d', elev=-150, azim=110)
arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)
# plot ox0y0z0 axes

for item in wrong_prediction_lr_indexes.T:
    # print(item)
    ax.text3D(x[:, 0].mean() + np.argmax(y[item]),
              x[:, 1].mean() + 1.5 - np.argmax(y[item]),
              x[:, 2].mean(), 'Ground Truth: '+iris_data_set.target_names[np.argmax(y[item])],
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    a = Arrow3D([x[:, 0].mean()+ np.argmax(y[item]), x[item, 0].mean()],
                [x[:, 1].mean()+ 1.5 -np.argmax(y[item]) , x[item, 1].mean()],
                [x[:, 2].mean(), x[item, 2].mean()], **arrow_prop_dict)
    ax.add_artist(a)

sc=ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y_pred_lr,
            edgecolor='k', label=[lt[0] for lt in labelTups])
plt.title('IRIS classification via Logistic Regression. The accuracy is: '+ str(1-wrong_prediction_lr_indexes.size/validation_lr_vector.size))
ax.set_xlabel(iris_data_set.feature_names[0])
ax.set_ylabel(iris_data_set.feature_names[1])
ax.set_zlabel(iris_data_set.feature_names[2])
colors = [sc.cmap(sc.norm(i)) for i in [0, 1, 2]]
custom_lines = [plt.Line2D([], [], ls="", marker='.',
                mec='k', mfc=c, mew=.1, ms=20) for c in colors]
ax.legend(custom_lines, [lt[0] for lt in labelTups],
          bbox_to_anchor=(0.3, .75))
fig.tight_layout()
plt.show()
