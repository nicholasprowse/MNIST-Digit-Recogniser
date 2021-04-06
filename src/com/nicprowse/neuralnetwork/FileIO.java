package com.nicprowse.neuralnetwork;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

import javax.imageio.ImageIO;

/**
 * Reads an IDX file. See http://yann.lecun.com/exdb/mnist/ for more information on the format
 * of IDX files. 
 * 
 * If the file has more than 1 dimension, then this splits up the file into a group
 * of matrices. The number of matrices is equal to the size of the first dimension. The matrix has 
 * dimension nx1, where n is the product of all the remaining dimensions. So, rather that retain the
 * dimensions, this flattens the data out into a column vector. For example, if the file has 
 * dimensions 10x6x4x5, it will be split into 10, 120x1 matrices. 
 * 
 * If the file has only one dimension, then it will convert each value into a 10x1 matrix where
 * the value is indicated by which element is 1. This means that all data values must be less than
 * or equal to 9. For example, if the byte was 4, it would be converted into the matrix
 * transpose({0, 0, 0, 0, 1, 0, 0, 0, 0, 0}).
 * @author Nicholas Prowse
 */
class IDXReader {
	
	public static final int LABEL = 0, DATA = 1, TRAINING = 0, TEST = 1;
	private InputStream stream;
	private String fileName;
	// The number of elements/matrices in the file
	private int length;
	// The number of bytes/data points in each element/matrix. If there is only 1
	// dimension in the IDX file, this will be 0
	private int dimension;
	// The index of the matrix that will be returned when nextMatrix is called
	private int currentMatrix = 0;
	
	/**
	 * Loads either the MNIST training data or MNIST test data as an array of DataPoint objects
	 * 
	 * @param type Determined if the loaded data is from the training set or the test set.
	 * must be one of IDXReader.TRAINING or IDXReader.TEST
	 * @return Array of DataPoint objects containing all the MNIST data of the given type
	 */
	public static DataPoint[] loadDataSet(int type, boolean augment) {
		IDXReader data   = new IDXReader(IDXReader.DATA, type),
				  labels = new IDXReader(IDXReader.LABEL, type);
		DataPoint[] dataSet = new DataPoint[Math.min(data.length, labels.length)];
		for(int i = 0; i < dataSet.length; i++) {
			dataSet[i] = new DataPoint(data.nextMatrix(augment), labels.nextMatrix());
		}
		data.close();
		labels.close();
		return dataSet;
	}
	
	/**
	 * Loads either the MNIST training data or MNIST test data as an array of DataPoint objects.
	 * Will only load a maximum of maxLength data points
	 * 
	 * @param type Determined if the loaded data is from the training set or the test set.
	 * must be one of IDXReader.TRAINING or IDXReader.TEST
	 * @param maxLength the maximum length of the returned data set.
	 * @return Array of DataPoint objects containing at most maxLength DataPoints from
	 *  the MNIST database of the given type
	 */
	public static DataPoint[] loadDataSet(int type, boolean augment, int maxLength) {
		IDXReader data   = new IDXReader(IDXReader.DATA, type),
				  labels = new IDXReader(IDXReader.LABEL, type);
		DataPoint[] dataSet = 
				new DataPoint[Math.min(maxLength, Math.min(data.length, labels.length))];
		for(int i = 0; i < dataSet.length; i++) 
			dataSet[i] = new DataPoint(data.nextMatrix(augment), labels.nextMatrix());
		
		data.close();
		labels.close();
		return dataSet;
	}
	
	
	/**
	 * Creates an IDXReader using one of the default files from the MNIST database.
	 * 
	 * @param labelOrData Indicates whether to load label data or image data. Must be one of
	 * IDXReader.LABEL or IDXReader.DATA
	 * @param trainOrTest Indicates whether to load training data or test data. Must be one of
	 * IDXReader.TRAINING or IDXReader.TEST
	 */
	public IDXReader(int labelOrData, int trainOrTest) {
		fileName = "data/" + (trainOrTest == TEST ? "test" : "training")
					 + "_" + (labelOrData == DATA ? "data" : "labels");
		stream = IDXReader.class.getResourceAsStream(fileName);
		initiateFile();
	}
	
	/**
	 * Creates an IDXReader from the given file
	 * @param file The file to load
	 */
	public IDXReader(File file) {
		fileName = file.getName();
		try {
			stream = new FileInputStream(file);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(0);
		}
		initiateFile();
	}
	
	/**
	 * Creates an IDXReader from the file determined by the given path
	 * @param path The path to the file to load
	 */
	public IDXReader(String path) {
		fileName = path;
		try {
			stream = new FileInputStream(path);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(0);
		}
		initiateFile();
	}
	
	/**
	 * Closes the InputStream thats reading the file
	 */
	public void close() {
		try {
			stream.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Reads the header of the IDX file and determines the dimension of the file.
	 * This confirms that the first 2 bytes are 0, that the third byte is 0x08, indicating
	 * that the data type in unsigned byte. It then determines the number of dimensions
	 * from the fourth byte. It then reads the size of each dimension. The size of each
	 * dimension is stored in a 4 byte MSB value stored sequentially after the first 4 bytes.
	 * From this, it sets the value of length, and dimension.
	 */
	private void initiateFile() {
		try {
			byte[] data = new byte[4];
			stream.read(data);
			// First 2 bytes must be 0
			if(data[0] != 0 || data[1] != 0) {
				showFormatError("The file '" + fileName + "' is corrupted.");
			}
			// Third byte is the data type. This only supports unsigned byte (0x80)
			if(data[2] != 0x08) {
				showFormatError("The file '" + fileName + "' does not have the correct type of "
						+ "unsigned byte. IDXReader only supports unsigned byte IDX files.");
			}
			// Fourth byte is the number of dimensions of the data structure
			int numDims = data[3];
			if(numDims == 0) {
				showFormatError("The file '" + fileName + "' has 0 dimensions. "
						+ "The data structures must have at least one dimension.");
			} else if(numDims > 3) {
				showFormatError("The file '" + fileName + "' has " + numDims + " dimensions. "
						+ "IDXReader only supports 1, 2 or 3 dimensions.");
			}
			// Next set of data is the size in each dimension. Each dimension is a 4 byte MSB value
			int[] dims = new int[numDims];
			data = new byte[numDims * 4];
			stream.read(data);
			for(int i = 0; i < numDims; i++) 
				for(int j = 0; j < 4; j++) 
					dims[i] = (dims[i] << 8) + Byte.toUnsignedInt(data[4*i + j]);
			
			length = dims[0];
			if(numDims == 1)
				dimension = 0;
			else {
				dimension = 1;
				for(int i = 1; i < dims.length; i++)
					dimension *= dims[i];
			}
		} catch (IOException e) {
			showFormatError("The file '" + fileName + "' is corrupted.");
		}
		
	}
	
	public boolean hasMore() {
		return currentMatrix < length;
	}
	
	/**
	 * Reads the next matrix from the file, and returns it. The returned matrix will always
	 * be a column vector containing the flattened data. If the file only had one dimension, 
	 * then each value is returned as a 10x1 vector where the value is determined by which 
	 * element is a 1 in the vector.
	 * 
	 * An extra 1 can optionally be added to the end of the matrix which is useful for 
	 * creating biases in neural networks. 
	 * 
	 * Note: Augmenting will not be applied if the data is 1 dimensional
	 * 
	 * @param augment boolean value indicating if the returned matrix should be augmented or
	 * not. An augmented matrix has an extra 1 appended to the start of the matrix
	 * 
	 * @return A Matrix containing the data read
	 */
	public Matrix nextMatrix(boolean augment) {
		try {
			if(dimension == 0) {
				double[] matrixData = new double[10];
				matrixData[stream.read()] = 1;
				currentMatrix++;
				return new Matrix(matrixData);
			} else {
				byte[] data = new byte[dimension];
				double[] pixels = new double[augment ? dimension+1 : dimension];
				stream.read(data);
				
				for(int i = 0; i < data.length; i++) 
					pixels[i] = (double)(Byte.toUnsignedInt(data[i]))/255;
				
				if(augment)
					pixels[pixels.length-1] = 1;
				currentMatrix++;
				return new Matrix(pixels);
			}
		} catch (IOException e) {
			showFormatError("The file '" + fileName + "' is corrupted.");
		}
		return null;
	}
	
	public Matrix nextMatrix() {
		return nextMatrix(false);
	}
	
	public Matrix[] getAllMatrices() {
		return null;
	}
	
	/**
	 * Shows an error and quits. The error should relate to the file being corrupted in
	 * some way. Also advises the user to re-download the file from the MNIST database
	 * @param error Error message to show
	 */
	private static void showFormatError(String error) {
		System.err.println(error + "\nPlease redownload the file from "
				+ "'http://yann.lecun.com/exdb/mnist/'");
		System.exit(0);
	}
	
	// Debugging method that allows me to see what an image looks like
	public static void saveImage(Matrix data, String name) throws IOException {
		BufferedImage bf = new BufferedImage(28, 28, BufferedImage.TYPE_4BYTE_ABGR);
		for(int x = 0; x < 28; x++)
			for(int y = 0; y < 28; y++) {
				int gray = (int)(data.getElement(y*28 + x, 0)*255);
				int rgb = (255 << 24) + (gray << 16) + (gray << 8) + gray;
				bf.setRGB(x, y, rgb);
			}
		
		ImageIO.write(bf, "png", 
				new File("/Users/nicholasprowse/Documents/Misc./Neural Network data/" + name + ".png"));
	}
	
}
