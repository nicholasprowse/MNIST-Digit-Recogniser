package com.nicprowse.neuralnetwork.gui;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.util.ArrayList;
import java.util.List;

import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.event.MouseInputListener;

import com.nicprowse.neuralnetwork.Matrix;
import com.nicprowse.neuralnetwork.Network;

public class GUI extends JFrame {

	private static final long serialVersionUID = 7926379525297805858L;
	private JLabel numberLabel;
	private DrawingPad pad;
	private Network net;

	public GUI(Network net) {
		super("Digit Recognizer");
		this.net = net;
		setLayout(new BoxLayout(getContentPane(), BoxLayout.X_AXIS));
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		getRootPane().setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
		pad = new DrawingPad(this);
		add(pad);
		JPanel rightPane = new JPanel();
		rightPane.setLayout(new BoxLayout(rightPane, BoxLayout.Y_AXIS));
		numberLabel = new JLabel(" ");
		numberLabel.setFont(new Font("Arial", Font.BOLD, 72));
		numberLabel.setAlignmentX(Component.CENTER_ALIGNMENT);
		numberLabel.setAlignmentY(Component.CENTER_ALIGNMENT);
		rightPane.add(numberLabel);
		JButton clear = new JButton("Clear");
		clear.setAlignmentX(Component.CENTER_ALIGNMENT);
		clear.setAlignmentY(Component.CENTER_ALIGNMENT);
		clear.addActionListener(e -> clear());
		rightPane.add(clear);
		rightPane.setBorder(BorderFactory.createEmptyBorder(0, 10, 0, 0));
		add(rightPane);
		
		pack();
		setLocationRelativeTo(null);
		setVisible(true);
		this.setResizable(false);
	}
	
	void guessNumber(BufferedImage img) {
		DataBuffer data = img.getData().getDataBuffer();
		double[] pixels = new double[data.getSize() + 1];
		for(int i = 0; i < pixels.length-1; i++)
			pixels[i] = 1 - (double)(data.getElem(i))/255.;
		pixels[pixels.length-1] = 1;
		Matrix result = net.feedForward(new Matrix(pixels));
		//result.transpose().print();
		int guess = result.getMaximumPosition()[0];
		numberLabel.setText(guess + "");
	}
	
	void clear() {
		numberLabel.setText(" ");
		pad.clear();
	}
	
}

class DrawingPad extends JPanel implements MouseInputListener {

	private static final long serialVersionUID = -7342252585079374516L;
	private static final int PIXEL_SIZE = 4, SIZE = PIXEL_SIZE * 28;
	private List<ArrayList<Integer>> x, y;
	private GUI parent;
	
	public DrawingPad(GUI parent) {
		setPreferredSize(new Dimension(SIZE, SIZE));
		addMouseMotionListener(this);
		addMouseListener(this);
		this.parent = parent;
		x = new ArrayList<ArrayList<Integer>>();
		y = new ArrayList<ArrayList<Integer>>();
	}
	
	public void clear() {
		x = new ArrayList<ArrayList<Integer>>();
		y = new ArrayList<ArrayList<Integer>>();
		repaint();
	}
	
	private BufferedImage getImage() {
		BufferedImage img = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
		Graphics2D g = img.createGraphics();
		g.setColor(Color.white);
		g.fillRect(0, 0, 28, 28);
		g.setColor(Color.black);
		g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		g.setStroke(new BasicStroke(3, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
		for(int i = 0; i < x.size(); i++) 
			g.drawPolyline(x.get(i).stream().mapToInt(j -> j).toArray(), 
					y.get(i).stream().mapToInt(j -> j).toArray(), x.get(i).size());
		
		return img;
	}
	
	public void paintComponent(Graphics cg) {
		cg.drawImage(getImage(), 0, 0, SIZE, SIZE, null);
	}

	@Override
	public void mouseDragged(MouseEvent e) {
		x.get(x.size()-1).add((int)Math.round((double)(e.getX()/PIXEL_SIZE)));
		y.get(y.size()-1).add((int)Math.round((double)(e.getY()/PIXEL_SIZE)));
		repaint();
	}

	@Override
	public void mouseMoved(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void mouseClicked(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void mousePressed(MouseEvent e) {
		x.add(new ArrayList<Integer>());
		y.add(new ArrayList<Integer>());
		x.get(x.size()-1).add((int)Math.round((double)(e.getX()/PIXEL_SIZE)));
		x.get(x.size()-1).add((int)Math.round((double)(e.getX()/PIXEL_SIZE)));
		y.get(y.size()-1).add((int)Math.round((double)(e.getY()/PIXEL_SIZE)));
		y.get(y.size()-1).add((int)Math.round((double)(e.getY()/PIXEL_SIZE)));
		repaint();
	}

	@Override
	public void mouseReleased(MouseEvent e) {
		parent.guessNumber(getImage());
	}

	@Override
	public void mouseEntered(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void mouseExited(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}
	
}
