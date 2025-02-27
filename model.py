import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Optimize memory for M3 Pro (8GB RAM)
if tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

# Data generators with stronger augmentations
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'Data/train',
    target_size=(224, 224),
    batch_size=8,  # Reduced batch size for efficiency
    class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    'Data/validation',
    target_size=(224, 224),
    batch_size=8,
    class_mode='binary'
)

# Build model
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze initial layers for transfer learning

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model with fewer epochs for quick testing
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator) // 2,
    epochs=5,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Convert to TFLite with FP16 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open('models/glaucoma_model_fp16.tflite', 'wb') as f:
    f.write(tflite_model)
print("✅ TFLite model saved: models/glaucoma_model_fp16.tflite")

# Save full model for Grad-CAM
model.save('models/glaucoma_model_efficientnet.h5')
print("✅ Full model saved: models/glaucoma_model_efficientnet.h5")
