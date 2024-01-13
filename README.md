# Project Overview: Gomoku
Welcome to the forefront of Gomoku artificial intelligence, where our project introduces a cutting-edge approach using a bespoke 2D Convolutional Neural Network (CNN). Gomoku, a classic board game demanding strategic finesse, has met its match in our meticulously designed AI model.

At the heart of our innovation is a Custom CNN Architecture, carefully crafted for optimal gameplay. The model encompasses five input layers and one output layer, boasting a 7x7 filter size. The Rectified Linear Activation Function (ReLU) injects non-linearity into all input layers, while the Sigmoid function gracefully maps output values within the range of 0 and 1. This architecture ensures efficient processing and strategic decision-making, enhancing the AI's ability to navigate the intricate game dynamics.

The foundation of our AI lies in the Standard version of the 2019 Gomoku Tournament game data obtained from Gomocup, a reputable platform for Gomoku AI competitions. Rigorous data cleaning processes were implemented, classifying players and removing extraneous information. Inputs and outputs were meticulously aligned to guarantee optimal player success. The resulting dataset, meticulously organized to represent game states on a 2D board with Max Player (X) as 1 and Min Player (O) as -1, became the cornerstone for our model training.

The data cleaning process paved the way for the creation of NumPy (.npz) files for all 2884 gameplays, subsequently utilized to train our Custom CNN. TensorFlow Keras, deployed on Google Colab, facilitated the training process. The architecture, featuring layers with 64, 128, 256, 128, and 64 filters, underwent ten epochs with a worker size of 30 and a batch size of 256, resulting in a finely tuned model.

Post-training, the model was compiled into a (.h5) file using the Adam optimizer, dynamically adjusting model weights during training. The binary cross-entropy loss function, chosen for its suitability to the binary classification nature of the problem, complemented the Sigmoid activation function in the output layer.

In the final stage, the trained model was integrated with TensorFlow Keras, predicting a 15x15 probability matrix based on the current game state. The argmax operation extracted the coordinates of the cell with the highest probability, orchestrating the final move in the Gomoku game.

The project's success is underscored by its favorable outcomes, consistently outperforming the baseline minimax model with a remarkable 25-victory average in 30 games. As we reflect on the project's evolution, we acknowledge the delicate balance between training duration (epochs) and the associated challenges. Future iterations may explore transitioning from a two-dimensional (2D) to a three-dimensional (3D) CNN architecture, along with the implementation of max-pooling techniques.

In conclusion, our Gomoku AI project stands as a testament to the convergence of sophisticated neural network architectures and strategic gameplay, offering a glimpse into the potential advancements in the realm of AI-driven board games.

References:

GomoCup - The Gomoku AI Tournament Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). Abadi, Martín, et al. TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. Software available from tensorflow.org. 2015.

I, Ankit Hiremath, am delighted to announce the successful completion of this innovative Gomoku AI project in collaboration with my esteemed teammates: Aditya Porwal, Titas Ghosh, Ameya Jajulwar, and Saad Shah.
