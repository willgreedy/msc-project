from helpers import load_model, save_model
import numpy as np

model = load_model('xor_sigmoid_lrx100_sp')

print(model)

_, layer = model.get_layers()[0]
_, layer2 = model.get_layers()[1]

feedforward_lr = layer.get_feedforward_learning_rate()
interneuron_lr = layer.get_interneuron_learning_rate()
predict_lr = layer.get_predict_learning_rate()

print(feedforward_lr)
print(interneuron_lr)
print(predict_lr)

orig_feedforward_lr1 = 0.0011875
orig_feedforward_lr2 = 0.0005
orig_predict_lr = 0.0011875
orig_interneuron_lr = 0.0059375

layer.set_feedforward_learning_rate(orig_feedforward_lr1)
layer2.set_feedforward_learning_rate(orig_feedforward_lr2)
layer.set_interneuron_learning_rate(orig_interneuron_lr)
layer.set_predict_learning_rate(orig_predict_lr)

interneuron_lr = layer.get_interneuron_learning_rate()
predict_lr = layer.get_predict_learning_rate()

print(feedforward_lr)
print(interneuron_lr)
print(predict_lr)

first_layer_feedforward_weights = layer.get_feedforward_weights()
feedforward_weights = layer2.get_feedforward_weights()
feedback_weights = layer.get_feedback_weights()

interneuron_weights = layer.get_interneuron_weights()
predict_weights = layer.get_predict_weights()

print(feedforward_weights.shape)
print(feedback_weights.shape)
print(interneuron_weights.shape)
print(predict_weights.shape)

layer.predict_weights = feedforward_weights.copy()
layer.interneuron_weights = -feedback_weights.copy()

#np.save("second_layer_feedforward_weights", feedforward_weights)
#np.save("first_layer_feedforward_weights", first_layer_feedforward_weights)

save_model('xor_sigmoid_lrx100_sp', model)

#a = np.random.uniform(-0.1, 0.1, (1000, 100))
#b = np.random.uniform(-0.1, 0.1, (1000, 100))

#scaled_dot_product = np.dot(a.flatten() / np.linalg.norm(a), b.T.flatten() / np.linalg.norm(b.T))
#if np.abs(scaled_dot_product) >= 1.0:
#    angle = 0.0
#else:
    #angle = np.degrees(np.arccos(scaled_dot_product))

#print(angle)