import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

np.random.seed(42)
tf.random.set_seed(42)

IMG_SIZE = 227 
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
NUM_CLASSES = 4

data_dir = "/Users/alaaabukmeil/Desktop/stat4012_project/brain-tumor-mri-dataset"
train_dir = os.path.join(data_dir, "Training")
test_dir = os.path.join(data_dir, "Testing")

val_split = 0.2 

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=val_split,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    color_mode='grayscale',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    color_mode='grayscale',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)


def create_alexnet(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    model = Sequential([
        Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='same', 
               activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        
        Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', 
               activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', 
               activation='relu'),
        BatchNormalization(),
        
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', 
               activation='relu'),
        BatchNormalization(),
        
        Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', 
               activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        
        Flatten(),
        
        Dense(4096, activation='relu'),
        Dropout(0.5),
        
        Dense(4096, activation='relu'),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def save_augmentation_examples(datagen, num_examples=5):
    x_batch, y_batch = next(train_generator)
    
    save_dir = "grayscale_augmentation_examples"
    os.makedirs(save_dir, exist_ok=True)
    
    sample_img = x_batch[0]
    
    plt.figure(figsize=(5, 5))
    plt.imshow(np.squeeze(sample_img), cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, "original.png"))
    plt.close()
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(np.squeeze(sample_img), cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    img_for_aug = np.expand_dims(sample_img, 0)
    aug_gen = datagen.flow(img_for_aug, batch_size=1)
    
    for i in range(5):
        aug_img = next(aug_gen)[0]
        plt.subplot(2, 3, i+2)
        plt.imshow(np.squeeze(aug_img), cmap='gray')
        plt.title(f"Augmentation {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "augmentation_grid.png"))
    plt.close()
    
    print(f"Saved augmentation examples to {save_dir}")

# save_augmentation_examples(train_datagen)
print(train_generator.samples)