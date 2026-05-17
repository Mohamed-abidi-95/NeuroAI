import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2

def build_efficientnet(input_shape=(128, 128, 3), num_classes=5,
                       backbone="efficientnetb0"):
           
    inputs = layers.Input(shape=input_shape)

    if backbone == "efficientnetb0":

        base_model = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
        )
        print("[✓] Backbone : EfficientNetB0 (ImageNet) — preprocessing intégré")
    else:

        x_preprocess = tf.keras.applications.resnet_v2.preprocess_input(inputs)
        base_model = ResNet50V2(
            include_top=False,
            weights="imagenet",
            input_tensor=x_preprocess,
        )
        print("[✓] Backbone : ResNet50V2 (ImageNet) — preprocessing intégré")

    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs,
                         name=f"TL_{backbone}_TumorStaging")
    return model, base_model

def compile_phase1(model, learning_rate=1e-3, loss_fn=None):
                                                   
    if loss_fn is None:
        loss_fn = "categorical_crossentropy"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy",
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    trainable = sum([tf.size(w).numpy() for w in model.trainable_variables])
    print(f"[Phase 1] Paramètres entraînables : {trainable:,}")
    return model

def unfreeze_for_finetuning(model, base_model, unfreeze_from_layer: int = -30,
                             learning_rate=1e-5, loss_fn=None):
           
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from_layer]:
        layer.trainable = False
    for layer in base_model.layers[unfreeze_from_layer:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False                           

    if loss_fn is None:
        loss_fn = "categorical_crossentropy"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy",
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    trainable = sum([tf.size(w).numpy() for w in model.trainable_variables])
    print(f"[Phase 2] Fine-Tuning — couches dégelées depuis index {unfreeze_from_layer}")
    print(f"          Paramètres entraînables : {trainable:,}")
    return model

if __name__ == "__main__":
    model, base = build_efficientnet()
    model = compile_phase1(model)
    model.summary(line_length=100)
