#### Mapping values to integer
when you have categorical values you want to convert it to numerical

tf.feature_column.categorical_column_with_vocabulary_list(
    key, vocabulary_list, dtype=None, default_value=-1, num_oov_buckets=0)

tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)

#### Create linear model
use a linear estimator to utilize the linear regression algorithm

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

#### Create DNNclassification model
It builds a DNN classifier with hidden units, feature columns and no. of classes you provided

classifier = tf.estimator.DNNClassifier(hidden_units, feature_columns,n_classes=none)

classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns, hidden_units = [30,10], n_classes = 3)

#### Training classifier Model
lambda is used to avoid creating inner function

classifier.train(input_fn = lambda: input_fn(train, train_y, training=True),steps=5000)

# Markov model distributions

tfp.distributions.HiddenMarkovModel(
    initial_distribution, transition_distribution, observation_distribution,
    num_steps, validate_args=False, allow_nan_stats=True,
    time_varying_transition_distribution=False,
    time_varying_observation_distribution=False, mask=None,
    name='HiddenMarkovModel'
)

#### Adding Convolutional layer and maxpooling layer
This allows us to extract the feature map. The MaxPooling with the stride of 2 allows to choose maximum value with the shift of 2.

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

#### Training the model
Allows to set the hyper-parameters of your choice.

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

fitting the model to the train images, test images, epochs, validation data

history = model.fit(train_images, train_labels, epochs=10, validation_data = (test_images, test_labels)  

#### Evaluate the model
Tests how the model performs on the test data

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

#### Data Agumentation(ImageDataGenerator)
It allows you generate more data from single image just by doing some operations like rotaion, flipping etc. 

datagen = ImageDataGenerator(
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')

#### Creating base model from pre-trained model
Creating base MobileNetV2 model  having input shape, include_top = False as we don't want to retrain the top layers and the weights are imagenet. 

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

#### Pad sequences
To make sequences of equal lengths, as we can not process with unequal length of sentences.

train_data = sequence.pad_sequences(train_data, MAXLEN)

#### Q-Learning
Using OpenAIs environment, use FrozenLake environment

env = gym.make("FrozenLake-v0") 






