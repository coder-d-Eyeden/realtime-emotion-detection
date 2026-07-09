import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(data_augmentation, input_shape=(48, 48, 1), num_classes=7):

    model = keras.Sequential(
        [
            # Input
            layers.Input(shape=input_shape),

            # Preprocessing
            data_augmentation,
            layers.Rescaling(1.0 / 255),

           
            # Block 1

            layers.Conv2D(
                32,
                (3, 3),
                padding="same",
                use_bias=False,
            ),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(32,(3, 3),padding="same",use_bias=False,),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),


            # Block 2
            layers.Conv2D(
                64,
                (3, 3),
                padding="same",
                use_bias=False,
            ),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(
                64,
                (3, 3),
                padding="same",
                use_bias=False,
            ),

            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
         
            # Block 3
            layers.Conv2D(
                128,
                (3, 3),
                padding="same",
                use_bias=False,
            ),

            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(
                128,
                (3, 3),
                padding="same",
                use_bias=False,
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.35),

            #Block 4

            layers.Conv2D(256,(3,3),padding="same",use_bias=False,),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(256,(3,3),padding="same",use_bias=False,),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.40),

          
            # Classification Head
            layers.GlobalAveragePooling2D(),
            layers.Dense(
                256,
                activation="relu",
            ),

            layers.Dropout(0.3),
            layers.Dense(
                num_classes,
                activation="softmax",
            ),

        ],
        name="EmotionCNN",
    )

    return model