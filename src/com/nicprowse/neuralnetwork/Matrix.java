package com.nicprowse.neuralnetwork;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

public class Matrix {
	private int rows, cols;
	private double[][] matrix;
	public Matrix(double[][] matrix) {
		this.matrix = matrix;
		this.rows = matrix.length;
		this.cols = matrix[0].length;
	}
	
	public Matrix(double[] matrix) {
		this.rows = matrix.length;
		this.cols = 1;
		this.matrix = new double[rows][cols];
		for(int i = 0; i < matrix.length; i++)
			this.matrix[i][0] = matrix[i];
	}
	
	public Matrix(int rows, int cols) {
		matrix = new double[rows][cols];
		this.rows = rows;
		this.cols = cols;
	}
	
	public Matrix(int length) {
		this(length, 1);
	}
	
	public static Matrix createRandomMatrix(int rows, int cols, double stdev) {
		double[][] matrix = new double[rows][cols];
		Random rand = new Random();
		for(int row = 0; row < matrix.length; row++)
			for(int col = 0; col < matrix[row].length; col++)
				matrix[row][col] = rand.nextGaussian() * stdev;
		
		return new Matrix(matrix);
	}
	
	public static Matrix createRandomVector(int length, double stdev) {
		return createRandomMatrix(length, 1, stdev);
	}
	
	public int getRows() {
		return rows;
	}
	
	public int getCols() {
		return cols;
	}
	
	public int[] shape() {
		return new int[] {rows, cols};
	}
	
	/**
	 * @return The total number of elements in this matrix. i.e. rows*cols
	 */
	public int size() {
		return rows*cols;
	}
	
	public double getElement(int row, int col) {
		return matrix[row][col];
	}
	
	public Matrix mult(Matrix m) {
		if(cols != m.rows) 
			System.err.println("Incompatible shapes for matrix multiplication\n"
					+ "\t(" + rows + ", " + cols + ") x (" + m.rows + ", " + m.cols + ")");
		
		double[][] matrix = new double[rows][m.cols];
		for(int row = 0; row < rows; row++)
			for(int col = 0; col < m.cols; col++)
				for(int i = 0; i < cols; i++)
					matrix[row][col] += this.matrix[row][i] * m.matrix[i][col];
		
		return new Matrix(matrix);
	}
	
	public Matrix mult(double k) {
		double[][] matrix = new double[rows][cols];
		for(int row = 0; row < rows; row++)
			for(int col = 0; col < cols; col++)
				matrix[row][col] = this.matrix[row][col] * k;
		
		return new Matrix(matrix);
	}
	
	public Matrix elMult(Matrix m) {
		if(cols != m.cols || rows != m.rows)
			System.err.println("Incompatible shapes for element-wise matrix multiplication\n"
					+ "\t(" + rows + ", " + cols + ") x (" + m.rows + ", " + m.cols + ")");
		
		double[][] matrix = new double[rows][cols];
		for(int row = 0; row < rows; row++)
			for(int col = 0; col < cols; col++)
				matrix[row][col] = this.matrix[row][col] * m.matrix[row][col];
		
		return new Matrix(matrix);
	}
	
	public Matrix add(Matrix m) {
		if(m == null)
			return this;
		if(cols != m.cols || rows != m.rows)
			System.err.println("Incompatible shapes for  matrix addition\n"
					+ "\t(" + rows + ", " + cols + ") + (" + m.rows + ", " + m.cols + ")");
		
		double[][] matrix = new double[rows][cols];
		for(int row = 0; row < rows; row++)
			for(int col = 0; col < cols; col++)
				matrix[row][col] = this.matrix[row][col] + m.matrix[row][col];
		
		return new Matrix(matrix);
	}
	
	public Matrix sub(Matrix m) {
		if(m == null)
			return this;
		if(cols != m.cols || rows != m.rows)
			System.err.println("Incompatible shapes for  matrix subtraction\n"
					+ "\t(" + rows + ", " + cols + ") - (" + m.rows + ", " + m.cols + ")");
		
		double[][] matrix = new double[rows][cols];
		for(int row = 0; row < rows; row++)
			for(int col = 0; col < cols; col++)
				matrix[row][col] = this.matrix[row][col] - m.matrix[row][col];
		
		return new Matrix(matrix);
	}
	
	public Matrix map(DoubleUnaryOperator f) {
		double[][] matrix = new double[rows][cols];
		for(int row = 0; row < rows; row++)
			for(int col = 0; col < cols; col++)
				matrix[row][col] = f.applyAsDouble(this.matrix[row][col]);
		
		return new Matrix(matrix);
	}
	
	public int[] getMaximumPosition() {
		int[] max = {0, 0};
		for(int row = 0; row < rows; row++)
			for(int col = 0; col < cols; col++)
				if(matrix[row][col] > matrix[max[0]][max[1]])
					max = new int[] {row, col};
		
		return max;
	}
	
	public Matrix transpose() {
		double[][] matrix = new double[cols][rows];
		for(int row = 0; row < rows; row++)
			for(int col = 0; col < cols; col++)
				matrix[col][row] = this.matrix[row][col];
		
		return new Matrix(matrix);
	}
	
	public static interface Function {
		public Matrix apply(Matrix m, int layer);
	}
	
	public void print() {
		for(int row = 0; row < rows; row++) {
			for(int col = 0; col < cols; col++) {
				String s = String.valueOf(Math.round(matrix[row][col] * 100)/100d);
				if(matrix[row][col] >= 0)
					s = " " + s;
				while(s.length() < 6)
					s += " ";
				System.out.print(s);
			}
			System.out.println();
		}
		System.out.println();
	}
}

class DataPoint {
	
	public Matrix data;
	public Matrix label;
	public DataPoint(Matrix data, Matrix label) {
		this.data = data;
		this.label = label;
	}
}
