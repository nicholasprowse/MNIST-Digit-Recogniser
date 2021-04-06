package com.nicprowse.neuralnetwork;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.util.function.DoubleUnaryOperator;

// TODO Add dropout to the network
// TODO Add artificial expansion of the training data
public class Network {
	
	private int layers;
	private Matrix[] weights;
	private DoubleUnaryOperator sigma = z -> 1/(1+Math.exp(-z)),
								dSigma = z -> sigma.applyAsDouble(z)*(1-sigma.applyAsDouble(z));
	
	/**
	 * Creates a neural network, with the given number of nodes in each layer. The number of
	 * arguments provided determines the number of layers in the neural network.
	 * The weights are initialised such that the standard deviation of any neurons weighted 
	 * input, z, is small (sqrt(3/2) = 1.22...). We need it to be small, because if its not,
	 * the neuron is saturated, resulting in slow learning.
	 * @param sizes The number of nodes in each layer. 
	 */
	public Network(int... sizes) {
		// Initialise random weights and biases
		layers = sizes.length - 1;
		weights = map((m, l) -> 
			Matrix.createRandomMatrix(sizes[l+1], sizes[l], 1/Math.sqrt(sizes[l])), new Matrix[layers]);
	}
	
	public Network(Matrix[] weights) {
		layers = weights.length;
		this.weights = weights;
		for(int i = 1; i < weights.length; i++) {
			if(weights[i].getCols() != weights[i-1].getRows()) {
				System.out.println("Invalid dimensions for weight matrices");
				System.exit(0);
			}
		}
	}
	
	/**
	 * Takes a matrix input, and determines the output of the neural network using the feed
	 * forward algorithm. The input must be a nx1 matrix where n is the number of nodes in the 
	 * first layer, and the output will be a mx1 matrix where m is the number of nodes in the
	 * final layer.
	 * @param input A Matrix of dimension nx1, where n is the number of nodes in the first layer.
	 * @return The output of the neural network. This is a matrix of dimension mx1 where m is the 
	 * number of nodes in the final layer
	 */
	public Matrix feedForward(Matrix input) {
		Matrix currLayer = input;
		for(int layer = 0; layer < layers; layer++) 
			currLayer = weights[layer].mult(currLayer).map(sigma);
		return currLayer;
	}
	
	/**
	 * Trains the neural network given some training data. The data is supplies as an array
	 * of DataPoint objects, which contain an input matrix and an expected output matrix, 
	 * named the data and the label respectively. This method will adjust the neural network
	 * so that it is able to produce the desired output for as many of the supplied data
	 * points as possible. While training, the test data is ignored, meaning that when we test
	 * the networks accuracy, we are testing against data that the neural network has never
	 * seen before.
	 * 
	 * The training is done by training the neural network on mini batches. The size of each 
	 * mini batch is determined by the batchSize parameter. The data set is split into these
	 * mini batches, and each one is trained. One time through the entire data set is called
	 * an epoch. The epochs parameter determines how many epochs are used. More epochs results
	 * in better results, but the network will reach a point where it cannot learn any more
	 * meaning more epochs won't make any difference. Each epoch, then data set is shuffled,
	 * allowing the batches to change in each epoch, allowing the network to learn slightly
	 * differently each time.
	 * 
	 * @param trainingData Array of DataPoint containing the data that the network will
	 * train with
	 * @param testData Array of DataPoint containing the data to test the network against
	 * @param epochs The number of times to train the network with the training data set
	 * @param batchSize The size of each training batch. Larger values will be slower, but
	 * will learn more effectively
	 * @param learningRate The learning rate. If this is too small, then the network will
	 * learn very slowly and require a lot of epochs to reach a reasonable accuracy. If it is 
	 * too large, then the network will change too much with each step, meaning it will 
	 * struggle to find the optimal values for the weights and biases
	 */
	public void train(DataPoint[] trainingData, DataPoint[] testData, int epochs, 
			int batchSize, double learningRate, double lambda) {
		printAccuracy(testData, 0);
		for(int epoch = 0; epoch < epochs; epoch++) {
			shuffle(trainingData);
			for(int batch = 0; batch < trainingData.length/batchSize; batch++)
				trainMiniBatch(trainingData, batch, batchSize, learningRate, lambda);
			printAccuracy(testData, epoch+1);
		}
	}
	
	/**
	 * Trains the neural network using the data from a single mini batch. The entire data set
	 * is passed into this method, but only the data from the batch is used. The mini batch
	 * is determined by the batchNumber and batchSize arguments.
	 * 
	 * This works by computing the derivative of the cost function, ∇C, for just the data points 
	 * in the mini batch. Since the cost function is the average of the cost function for
	 * each data point, we simply compute the derivative for each data point (using backPropagate)
	 * and add them all together. Then, at the end we divide by the batch size so it is an average.
	 * 
	 * We then subtract this from the current parameters. The reason this works is because ∇C always
	 * points in the direction of maximum increase of C, meaning that -∇C always points in the 
	 * direction of maximum decrease of C. Thus, if we always move params by -∇C we will reduce the
	 * cost. We scale ∇C by the learning rate so we can control how fast ∇C changes
	 * 
	 * When we subtract the derivatives, we also scale the weights down by a small factor determined
	 * by lambda. This results in large weights decaying down to smaller values. The reason this is 
	 * done is to prevent weights becoming too large. Large weights are an issue, because when a 
	 * weight is large, subtracting the derivative only has a small relative effect, meaning that 
	 * learning is slower that it would be with small weights.
	 * 
	 * @param data The entire data set that the network is being trained on
	 * @param batchNumber an integer determining which batch this is. This is how the location
	 * of the match within data is determined
	 * @param batchSize the size of the batch
	 * @param learningRate the learning rate
	 * @param lambda the regularization parameter. This determines how quickly the weights decay
	 */
	private void trainMiniBatch(DataPoint[] data, int batchNumber, int batchSize, 
			double learningRate, double lambda) {
		Matrix[] nabla = new Matrix[layers];
		
		for(int i = batchSize*batchNumber; i < batchSize*(batchNumber+1); i++) {
			Matrix[] dNabla = backPropagate(data[i]);
			map((m, l) -> dNabla[l].add(nabla[l]), nabla);
		}
		
		map((m, l) -> m.mult(1-learningRate*lambda/data.length)
				.sub(nabla[l].mult(learningRate/batchSize)), weights);
	}
	
	private void shuffle(DataPoint[] dataSet) {
		for(int i = dataSet.length - 1; i > 0; i--) {
			int r = (int)(Math.random() * (i+1));
			DataPoint temp = dataSet[i];
			dataSet[i] = dataSet[r];
			dataSet[r] = temp;
		}
	}
	
	/**
	 * Computes the derivative of the cost function of the given data point with respect 
	 * to all of the weights and biases. This is done using the back propagation algorithm.
	 * First, the data is fed through the network using the feed forward algorithm. At each
	 * stage the weighed input, z, and the activation, a, is saved using the formulae
	 * 
	 * z[l] = w[l]a[l-1] + b[l]									(1)
	 * a[l] = σ(z[l])											(2)
	 * 
	 * We then perform back propagation to compute δ (d in code). δ is defined as the 
	 * derivative of the cost function with respect to the weighted inputs, z. This is 
	 * calculated using these two formulae
	 * 
	 * δ[L] = dC/da[L] * σ'(z[L]) = (y - a[L])					(3)
	 * δ[l] = Tr(w[l+1])δ[l+1] * σ'(z[l])						(4)
	 * 
	 * Where L indicates the final layer, Tr is the transpose, * indicates element-wise 
	 * multiplication and y is the expected output (the label). Here, we are using the
	 * cross entropy cost function, which is designed so that the derivative cancels out
	 * the σ'(z[L]) term. This term is problematic because if z[L] is large in absolute
	 * terms, then σ'(z[L]) is very small, meaning δ[L] is very small. This results in
	 * slower learning, since the amount that the weights/biases are changed is directly
	 * dependent on this value. It is impossible to remove the σ'(z[l]) term for the 
	 * inner layer neurons, but we can mitigate its effects by initialising the weights 
	 * biases such that it is unlikely that any neurons are saturated (large absolute z)
	 * to start with
	 * 
	 * C = -(y ln(a) + (1-y) ln(1-a))
	 * 
	 * Finally, we can relate δ to the derivatives of the cost function with respect to 
	 * the weights and biases
	 * 
	 * dC/dw[l] = a[l-1]δ[l]									(5)
	 * 
	 * All of these formulae can be derived relatively easily from the definitions, and the 
	 * feed forward formulae. It is important to note that my weights and biases are offset
	 * by one, so the indices will not match in the code.
	 * @param data A DataPoint containing the data and label for a single training data point
	 * @return A NetworkParameters object containing the derivatives of the cost function of
	 * the supplied data with respect to every weight and bias
	 */
	private Matrix[] backPropagate(DataPoint data) {
		Matrix[] a = new Matrix[layers+1];
		Matrix[] z = new Matrix[layers+1];
		Matrix[] d = new Matrix[layers+1];
		a[0] = data.data;
		// Feed forward
		for(int layer = 0; layer < layers; layer++) {
			z[layer+1] = weights[layer].mult(a[layer]);		// Equation 1
			a[layer+1] = z[layer+1].map(sigma);				// Equation 2
		}
		// Back propagate
		d[d.length-1] = a[layers].sub(data.label);			// Equation 3
		for(int layer = layers-1; layer > 0; layer--)
			// Equation 4
			d[layer] = weights[layer].transpose().mult(d[layer+1]).elMult(z[layer].map(dSigma));
		// Equation 5
		return map((m, l) -> d[l+1].mult(a[l].transpose()), new Matrix[layers]);
	}
	
	public void printAccuracy(DataPoint[] testData, int epoch) {
		double[] correct = new double[10];
		double[] total = new double[10];
		double totalCorrect = 0;
		for(int i = 0; i < testData.length; i++) {
			Matrix result = feedForward(testData[i].data);
			int answer = result.getMaximumPosition()[0];
			if(testData[i].label.getElement(answer, 0) == 1) {
				totalCorrect++;
				correct[answer]++;
			}
			total[testData[i].label.getMaximumPosition()[0]]++;
		}
		
		System.out.print("Epoch " + epoch + ": Accuracy = {");
		for(int i = 0; i < total.length; i++) {
			double percent = Math.round(correct[i]*10000/total[i])/100d;
			System.out.print(String.format("%d: %04.2f%%, ", i, correct[i]*100/total[i]));
			//System.out.print(i + ": " + percent + "%, ");
		}
		totalCorrect = Math.round(totalCorrect*10000/testData.length)/100d;
		System.out.println("Total: " + totalCorrect + "%}");
	}
	
	private Matrix[] map(Matrix.Function f, Matrix[] matrix) {
		for(int layer = 0; layer < matrix.length; layer++)
			matrix[layer] = f.apply(matrix[layer], layer);
		
		return matrix;
	}
	
	/**
	 * Saves the network as a binary file. The file has the following format
	 * - 6 validation bytes that are the ascii values for the word 'neural'
	 * - 4 byte integer indicating the number of layers
	 * - sequence of 4 byte integers indicating the number of neurons in each layer
	 * - sequence of 8 byte double values containing all the weights of the network. 
	 * The weights are ordered from the first to last layer, and each layer is ordered
	 * row by row in the weights matrix.
	 */
	public void saveNetwork() {
		int fileLength = 10 + (weights.length + 1) * 4;
		for(Matrix w : weights)
			fileLength += w.size() * 8;
		byte[] data = new byte[fileLength];
		
		// Validation string
		String validation = "neural";
		for(int i = 0; i < validation.length(); i++)
			data[i] = (byte)validation.charAt(i);
		
		// Number of layers
		for(int i = 0; i < 4; i++)
			data[6 + i] = (byte)((weights.length+1) >> (8*(3 - i)));
		
		// Size of each layer
		for(int j = 0; j < weights.length; j++)
			for(int i = 0; i < 4; i++)
				data[10 + 4*j + i] = (byte)(weights[j].getCols() >> (8*(3 - i)));
		
		for(int i = 0; i < 4; i++)
			data[10 + weights.length * 4 + i] = 
				(byte)(weights[weights.length-1].getRows() >> (8*(3 - i)));
		
		// weights data
		int index = 10 + (weights.length + 1)*4;
		for(Matrix w : weights) {
			for(int row = 0; row < w.getRows(); row++) {
				for(int col = 0; col < w.getCols(); col++) {
					long weight = Double.doubleToLongBits(w.getElement(row, col));
					for(int i = 0; i < 8; i++)
						data[index + i] = (byte)(weight >> (8*(7 - i)));
					index += 8;
				}
			}
		}
		try {
			File file = Paths.get(Network.class.getResource("data/network").toURI()).toFile();
			OutputStream stream = new FileOutputStream(file);
			stream.write(data);
			stream.close();
		} catch (IOException | URISyntaxException e) {
			e.printStackTrace();
		}
	}
	
	public static Network loadNetwork() {
		byte[] data = null;
		try {
			InputStream stream = Network.class.getResourceAsStream("data/network");
			data = stream.readAllBytes();
			stream.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		// Check validation string
		String validation = "neural";
		for(int i = 0; i < validation.length(); i++)
			if(data[i] != (byte)validation.charAt(i)) {
				System.err.println("The saved neural network is corrupted, and cannot be loaded");
				System.exit(0);
			}
		
		// Get number of layers
		int layers = 0;
		for(int i = 0; i < 4; i++)
			layers = (layers << 8) + data[6+i];
		
		// Get each layers sizes
		int[] layerSizes = new int[layers];
		for(int j = 0; j < layers; j++)
			for(int i = 0; i < 4; i++)
				layerSizes[j] = (layerSizes[j] << 8) + data[10 + 4*j + i];
		
		// Get all the weights
		int index = 10 + layers*4;
		Matrix[] weights = new Matrix[layers-1];
		for(int m = 0; m < weights.length; m++) {
			double[][] w = new double[layerSizes[m+1]][layerSizes[m]];
			for(int row = 0; row < w.length; row++) {
				for(int col = 0; col < w[row].length; col++) {
					long weight = 0;
					for(int i = 0; i < 8; i++) 	
						weight = (weight << 8) + Byte.toUnsignedLong(data[index+i]);
					w[row][col] = Double.longBitsToDouble(weight);
					index += 8;
				}
			}
			weights[m] = new Matrix(w);
		}

		return new Network(weights);
	}
	
	public void printWeights() {
		for(int i =0; i < weights.length; i++) {
			System.out.println("-------------------- Layer: " + i + " --------------------");
			for(int row = 0; row < weights[i].getRows(); row++) {
				for(int col = 0; col < weights[i].getCols(); col++) {
					System.out.print(weights[i].getElement(row, col) + "  ");
				}
				System.out.println();
				
			}
		}
	}
}
