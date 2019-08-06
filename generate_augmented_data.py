from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = 'data/caps and shoes/train'
valid_path = 'data/caps and shoes/val'
test_path = 'data/caps and shoes/test'

GEN_TRAIN_IMAGE_COUNT = 400
GEN_VAL_IMAGE_COUNT = 200

datagen = ImageDataGenerator(rotation_range=30,
				   zoom_range=0.15,
				   width_shift_range=0.2,
				   height_shift_range=0.2,
				   shear_range=0.15,
				   horizontal_flip=True,
				   fill_mode="nearest")
			
train_cap_gen = datagen.flow_from_directory(train_path, target_size=(112,112), classes=['cap'], color_mode = 'grayscale', batch_size=1, save_to_dir='data/caps and shoes generated/train/cap/', save_prefix="gen_cap_image", save_format="jpeg")
train_shoes_gen = datagen.flow_from_directory(train_path, target_size=(112,112), classes=['shoes'], color_mode = 'grayscale', batch_size=1, save_to_dir='data/caps and shoes generated/train/shoes/', save_prefix="gen_shoes_image", save_format="jpeg")

val_cap_gen = datagen.flow_from_directory(valid_path, target_size=(112,112), classes=['cap'], color_mode = 'grayscale', batch_size=1, save_to_dir='data/caps and shoes generated/val/cap/', save_prefix="gen_cap_image", save_format="jpeg")
val_shoes_gen = datagen.flow_from_directory(valid_path, target_size=(112,112), classes=['shoes'], color_mode = 'grayscale', batch_size=1, save_to_dir='data/caps and shoes generated/val/shoes/', save_prefix="gen_shoes_image", save_format="jpeg")

#test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(112,112), classes=['cap', 'shoes'], batch_size=1)

total = 0

print("generating training images...")

for cap,shoes in zip(train_cap_gen,train_shoes_gen):
  total += 1
  if total == GEN_TRAIN_IMAGE_COUNT:
    break

total = 0

print("generating validation images...")

for cap,shoes in zip(val_cap_gen,val_shoes_gen):
  total += 1
  if total == GEN_VAL_IMAGE_COUNT:
    break


