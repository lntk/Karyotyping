from keras.applications.xception import Xception
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras_contrib.applications import ResNet18
from keras_retinanet.bin.train import *


def khang1_regression(input_shape, num_output):
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(
        Conv2D(24, 5, 5, border_mode='same', init='he_normal',
               input_shape=input_shape,
               dim_ordering='tf'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))
    model.add(Conv2D(36, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))
    model.add(Conv2D(48, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))
    model.add(Conv2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))
    model.add(Conv2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(90, activation='relu'))
    model.add(Dense(num_output))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

    return model


def resnet18_regression(input_shape, num_output):
    base_model = ResNet18(input_shape, num_output)
    x = base_model.layers[-2].output
    skeleton = Dense(num_output, activation='linear')(x)

    model = Model(inputs=base_model.input, outputs=skeleton)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

    return model


def xception_regression(input_shape, num_output):
    base_model = Xception(include_top=True, weights=None, input_shape=input_shape, classes=num_output)
    x = base_model.layers[-2].output
    skeleton = Dense(num_output, activation='linear')(x)

    model = Model(inputs=base_model.input, outputs=skeleton)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

    return model


def retinanet_regression(annotations_file, class_mapping_file):
    # parse arguments
    args = ["csv", annotations_file, class_mapping_file]
    args = parse_args(args)

    # create object that stores backbone information
    backbone = models.backbone(args.backbone)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # create the generators
    train_generator, validation_generator = create_generators(args, backbone.preprocess_image)

    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model = models.load_model(args.snapshot, backbone_name=args.backbone)
        training_model = model
        anchor_params = None
        if args.config and 'anchor_parameters' in args.config:
            anchor_params = parse_anchor_parameters(args.config)
        prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)
    else:
        weights = args.weights
        # default to imagenet if nothing else is specified
        if weights is None and args.imagenet_weights:
            weights = backbone.download_imagenet()

        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone,
            lr=args.lr,
            config=args.config
        )

    # print model summary
    model.summary()

    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'vgg' in args.backbone or 'densenet' in args.backbone:
        train_generator.compute_shapes = make_shapes_callback(model)
        if validation_generator:
            validation_generator.compute_shapes = train_generator.compute_shapes

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        args,
    )

    # Use multiprocessing if workers > 0
    if args.workers > 0:
        use_multiprocessing = True
    else:
        use_multiprocessing = False

    X, y = train_generator.__getitem__(0)
    print(X.shape)
    print(len(y))
    print(X)
    print(y)

    # # start training
    # training_model.fit_generator(
    #     generator=train_generator,
    #     steps_per_epoch=args.steps,
    #     epochs=args.epochs,
    #     verbose=1,
    #     callbacks=callbacks,
    #     workers=args.workers,
    #     use_multiprocessing=use_multiprocessing,
    #     max_queue_size=args.max_queue_size
    # )


wrong_annotation = ["xx_karyotype_038_0.bmp",
                    "xx_karyotype_040_0.bmp",
                    "xx_karyotype_049_1.bmp",
                    "xx_karyotype_055_1.bmp",
                    "xx_karyotype_124_0.bmp",
                    "xy_karyotype_032_1.bmp",
                    "xy_karyotype_051_0.bmp",
                    "xy_karyotype_075_1.bmp",
                    "xy_karyotype_162_1.bmp",
                    "xy_karyotype_203_0.bmp",
                    "xy_karyotype_212_0.bmp",
                    "xy_karyotype_227_1.bmp"]
