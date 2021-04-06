package com.nicprowse.neuralnetwork;

import com.nicprowse.neuralnetwork.gui.GUI;

public class Main {

	public static void main(String[] args) throws Exception {
		// Uncomment this to train for 30 epochs and save the resultant network
		/* 
		DataPoint[] trainingData = IDXReader.loadDataSet(IDXReader.TRAINING, true);
		DataPoint[] testData = IDXReader.loadDataSet(IDXReader.TEST, true);
		System.out.println("Loaded Data");
		Network net = new Network(785, 101, 10);
		net.train(trainingData, testData, 30, 10, 0.5, 5.0);
		net.saveNetwork();
		*/
		
		DataPoint[] testData = IDXReader.loadDataSet(IDXReader.TEST, true);		
		Network net = Network.loadNetwork();
		net.printAccuracy(testData, 30);
		new GUI(net);
	}

}