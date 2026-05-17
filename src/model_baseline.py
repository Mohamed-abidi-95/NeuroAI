import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_baseline_cnn(input_shape=(128, 128, 3), num_classes=5):
           
    model = models.Sequential(name="CNN_Baseline_TumorStaging")

    model.add(layers.Input(shape=input_shape))

    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.30))

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(num_classes, activation="softmax"))

    return model

def compile_model(model, learning_rate=1e-3, loss_fn=None):
           
    if loss_fn is None:
        loss_fn = "categorical_crossentropy"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[
                      ,
            tf.keras.metrics.Recall(name="recall"),                    
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.AUC(name="auc"),                      
        ],
    )
    return model

if __name__ == "__main__":
    model = build_baseline_cnn()
    model = compile_model(model)
    model.summary()
    print(f"\n[✓] Paramètres totaux : {model.count_params():,}")
