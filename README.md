# Deep_NN
Deep Neural Network implemented in **Python** from scratch :smile:
<br>
This currently supports:
- **Deep Neural Networks** with fully connected layers and non-linearities
- **Classification** (Logistic and softmax) and **Regression** (L2)
- Uses **Gradient Descent** optimizer

### Modules
This actually contains four Modules:
- Layer
- Loss
- NN_model
- ProgressBar

Each of these above mentioned Modules contains a class whose name is same as module name

### Example Code
Lets create a **3-layer Neural Network** (two Hidden layers and a output layer)

```python
from NN_model import *

3_layer_NN = NN_model(input_size=64, layers=[
                    Layer(units=32, activation="relu"),
                    Layer(units=16, activation="relu"),
                    Layer(units=5, activation="softmax")],
                    loss_type="categorical_crossentropy",
                    print_loss="True"
                    )
```

So this model takes in input of shape (64, *m*) where *m* is number of training examples. And each layer in the Neural Network belongs to <code>Layer</code> class which is implemented Layer module. Each layer takes two mandatory inputs *units*, *activation* where activation can take one of the types <code>relu</code>, <code>sigmoid</code>,  <code>softmax</code>

And there are three types of losses: <code>mse</code>, <code>binary\_crossentropy</code>, <code>categorical\_crossentropy</code> and <code>print_loss</code>
to print loss after each epoch.

Now lets fit the network,

```python
3_layer_NN.fit(X, Y, epochs=1000, learning_rate=0.01)
```

So to predict and evaluate on some examples,

```python
prediction = 3_layer_NN.predict(example)
3_layer_NN.evaluate(X_val, Y_val)
```

Finally, to view the details of network.

```python    
3_layer_NN.summary()
```
    
