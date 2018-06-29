This is a simple implementaion of OCR with backpropagating neural network.
Input: PNG image.
Output: Text file -> "output_final.txt"

Scikit learn library is used for the neural network.
The training data consists of alphabets and digits of fonts: Arial and Times New Roman.
Training Data is first classified to reduce the training time.
Given Testcases can be used to test the program.
The program creates obj files, for the trained instances of the neural networks for different categories
of the training data(Each category of the trianing dat has different neural network).
These obj files can be then read by modifying the main program, and negates the need for training the 
networks each time the program is run.