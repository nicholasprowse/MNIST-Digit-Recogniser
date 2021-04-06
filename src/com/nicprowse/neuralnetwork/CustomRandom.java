package com.nicprowse.neuralnetwork;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class CustomRandom {

	private boolean hasNextGaussian = false;
	private double nextGaussian = 0;
	private BufferedReader file;
	
	public CustomRandom() {
		try {
			file = new BufferedReader(new FileReader(
					new File("/Users/nicholasprowse/Desktop/rand.txt")));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public double nextUniform() {
		try {
			return Double.parseDouble(file.readLine());
		} catch (NumberFormatException | IOException e) {
			e.printStackTrace();
		}
		return 0;
	}
	
	public double nextGaussian() {
		if (hasNextGaussian) {
			hasNextGaussian = false;
            return nextGaussian;
        } else {
            double v1, v2, s;
            do {
                v1 = 2 * nextUniform() - 1; // between -1 and 1
                v2 = 2 * nextUniform() - 1; // between -1 and 1
                s = v1 * v1 + v2 * v2;
            } while (s >= 1 || s == 0);
            double multiplier = StrictMath.sqrt(-2 * StrictMath.log(s)/s);
            nextGaussian = v2 * multiplier;
            hasNextGaussian = true;
            return v1 * multiplier;
        }
	}
	
}
