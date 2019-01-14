from script.model import model_lib

model = model_lib.xception_regression(input_shape=(512, 512, 1), num_output=2)
model.summary()
