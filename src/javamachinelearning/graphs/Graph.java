package javamachinelearning.graphs;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;

import javax.imageio.ImageIO;

import javamachinelearning.utils.Utils;

public class Graph{
	private BufferedImage graph;
	private Graphics2D graphics;
	private ArrayList<Point> points = new ArrayList<>();
	private ArrayList<Line> lines = new ArrayList<>();
	private ArrayList<LineGraph> lineGraphs = new ArrayList<>();
	private int width;
	private int height;
	private int xTicks;
	private int yTicks;
	private int padding;
	private String xLabel;
	private String yLabel;
	private ColorFunction colorFunction;
	private boolean customScale = false;
	private double minX;
	private double maxX;
	private double minY;
	private double maxY;
	
	public Graph(){
		this(500, 500);
	}
	
	public Graph(int width, int height){
		this(width, height, null, null, null, null);
	}
	
	public Graph(int width, int height, String xLabel, String yLabel){
		this(width, height, 10, 10, 100, xLabel, yLabel, null, null, null, null);
	}
	
	public Graph(ColorFunction colorFunction){
		this(500, 500, colorFunction);
	}
	
	public Graph(int width, int height, ColorFunction colorFunction){
		this(width, height, null, null, null, colorFunction);
	}
	
	public Graph(int width, int height, double[] xData, double[] yData, Color[] cData, ColorFunction colorFunction){
		this(width, height, "x-axis", "y-axis", xData, yData, cData, colorFunction);
	}
	
	public Graph(int width, int height, String xLabel, String yLabel, double[] xData, double[] yData, Color[] cData, ColorFunction colorFunction){
		this(width, height, 10, 10, 100, xLabel, yLabel, xData, yData, cData, colorFunction);
	}
	
	public Graph(int width, int height, int xTicks, int yTicks, int padding, String xLabel, String yLabel, double[] xData, double[] yData, Color[] cData, ColorFunction colorFunction){
		this.graph = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
		this.graphics = this.graph.createGraphics();
		this.width = width;
		this.height = height;
		this.xTicks = xTicks + 1;
		this.yTicks = yTicks + 1;
		this.padding = padding;
		this.xLabel = xLabel;
		this.yLabel = yLabel;
		this.colorFunction = colorFunction;
		
		if(xData != null && yData != null){
			for(int i = 0; i < xData.length; i++){
				if(cData == null || cData.length < xData.length)
					this.points.add(new Point(xData[i], yData[i]));
				else
					this.points.add(new Point(xData[i], yData[i], cData[i]));
			}
		}
	}
	
	public void useCustomScale(double minX, double maxX, double minY, double maxY){
		this.minX = minX;
		this.maxX = maxX;
		this.minY = minY;
		this.maxY = maxY;
		this.customScale = true;
	}
	
	public void usePointScale(){
		this.customScale = false;
	}
	
	public void draw(){
		// find graph range
		double xMax = Double.MIN_VALUE;
		double xMin = Double.MAX_VALUE;
		double yMax = Double.MIN_VALUE;
		double yMin = Double.MAX_VALUE;
		if(customScale){
			xMax = maxX;
			xMin = minX;
			yMax = maxY;
			yMin = minY;
		}else{
			for(int i = 0; i < points.size(); i++){
				xMax = Math.max(xMax, points.get(i).getX());
				xMin = Math.min(xMin, points.get(i).getX());
				yMax = Math.max(yMax, points.get(i).getY());
				yMin = Math.min(yMin, points.get(i).getY());
			}
			for(int i = 0; i < lineGraphs.size(); i++){
				ArrayList<Point> arr = lineGraphs.get(i).getPoints();
				for(int j = 0; j < arr.size(); j++){
					xMax = Math.max(xMax, arr.get(j).getX());
					xMin = Math.min(xMin, arr.get(j).getX());
					yMax = Math.max(yMax, arr.get(j).getY());
					yMin = Math.min(yMin, arr.get(j).getY());
				}
			}
			if(xMax == Double.MIN_VALUE)
				xMax = 10;
			if(xMin == Double.MAX_VALUE)
				xMin = 0;
			if(yMax == Double.MIN_VALUE)
				yMax = 10;
			if(yMin == Double.MAX_VALUE)
				yMin = 0;
			if(xMax - xMin > yMax - yMin){
				double diff = xMax - xMin - yMax + yMin;
				yMin -= diff / 2;
				yMax += diff / 2;
			}else{
				double diff = yMax - yMin - xMax + xMin;
				xMin -= diff / 2;
				xMax += diff / 2;
			}
		}
		
		if(colorFunction != null){
			int xSize = 500;
			int ySize = 500;
			for(int i = 0; i < xSize; i++){
				for(int j = 0; j < ySize; j++){
					graphics.setColor(colorFunction.getColor(
							xMin + i / (double)xSize * (xMax - xMin),
							yMin + j / (double)ySize * (yMax - yMin)));
					graphics.fillRect(
							(int)(padding * 2 + i * (width - padding * 3) / (double)xSize),
							(int)(height - padding * 2 - (j + 1) * ((height - padding * 3) / (double)ySize)),
							(int)((width - padding * 3) / (double)xSize),
							(int)((height - padding * 3) / (double)ySize));
				}
			}
		}
		
		// x and y axis
		graphics.setColor(Color.black);
		graphics.setStroke(new BasicStroke(3));
		graphics.drawLine(padding * 2, height - padding * 2, width - padding, height - padding * 2);
		graphics.drawLine(padding * 2, height - padding * 2, padding * 2, padding);
		
		// x and y axis labels
		graphics.setFont(graphics.getFont().deriveFont(50.0f));
		graphics.drawString(xLabel, width / 2 - 5 * xLabel.length(), height - padding);
		AffineTransform oldTransform = graphics.getTransform();
		graphics.translate(padding, height / 2 + 5 * yLabel.length());
		graphics.rotate(Math.toRadians(-90.0));
		graphics.drawString(yLabel, 0, 0);
		graphics.setTransform(oldTransform);
		
		// draw tick marks
		graphics.setFont(graphics.getFont().deriveFont(25.0f));
		int xTickSpacing = (width - padding * 3) / (xTicks - 1);
		int yTickSpacing = (height - padding * 3) / (yTicks - 1);
		for(int i = 0; i < xTicks; i++){
			graphics.drawLine(
					padding * 2 + xTickSpacing * i,
					height - padding * 2,
					padding * 2 + xTickSpacing * i,
					height - padding * 2 + 10);
			graphics.drawString(
					Utils.shorterFormat(xMin + (xMax - xMin) / (xTicks - 1) * i),
					padding * 2 + xTickSpacing * i - 7,
					height - padding * 2 + 40);
		}
		for(int i = 0; i < yTicks; i++){
			graphics.drawLine(
					padding * 2,
					height - padding * 2 - yTickSpacing * i,
					padding * 2 - 10,
					height - padding * 2 - yTickSpacing * i);
			graphics.drawString(
					Utils.shorterFormat(yMin + (yMax - yMin) / (yTicks - 1) * i),
					padding * 2 - 70,
					height - padding * 2 - yTickSpacing * i + 10);
		}
		
		// draw points
		for(int i = 0; i < points.size(); i++){
			graphics.setColor(Color.black);
			graphics.fillOval(
					padding * 2 + (int)((points.get(i).getX() - xMin) / (xMax - xMin) * (width - padding * 3)) - 10,
					(height - padding * 2) - (int)((points.get(i).getY() - yMin) / (yMax - yMin) * (height - padding * 3)) - 10, 20, 20);
			graphics.setColor(points.get(i).getColor());
			graphics.fillOval(
					padding * 2 + (int)((points.get(i).getX() - xMin) / (xMax - xMin) * (width - padding * 3)) - 8,
					(height - padding * 2) - (int)((points.get(i).getY() - yMin) / (yMax - yMin) * (height - padding * 3)) - 8, 16, 16);
		}
		
		// draw line
		for(int i = 0; i < lines.size(); i++){
			graphics.setColor(lines.get(i).getColor());
			double y1 = lines.get(i).getM() * xMin + lines.get(i).getB();
			double y2 = lines.get(i).getM() * xMax + lines.get(i).getB();
			graphics.drawLine(
					padding * 2,
					(height - padding * 2) - (int)((y1 - yMin) / (yMax - yMin) * (height - padding * 3)),
					padding * 2 + width - padding * 3,
					(height - padding * 2) - (int)((y2 - yMin) / (yMax - yMin) * (height - padding * 3)));
		}
		
		// draw line graphs
		for(int i = 0; i < lineGraphs.size(); i++){
			Point prev = null;
			LineGraph g = lineGraphs.get(i);
			for(int j = 0; j < g.getPoints().size(); j++){
				Point p = g.getPoints().get(j);
				graphics.setColor(Color.black);
				graphics.fillOval(
						padding * 2 + (int)((p.getX() - xMin) / (xMax - xMin) * (width - padding * 3)) - 10,
						(height - padding * 2) - (int)((p.getY() - yMin) / (yMax - yMin) * (height - padding * 3)) - 10, 20, 20);
				graphics.setColor(p.getColor());
				graphics.fillOval(
						padding * 2 + (int)((p.getX() - xMin) / (xMax - xMin) * (width - padding * 3)) - 8,
						(height - padding * 2) - (int)((p.getY() - yMin) / (yMax - yMin) * (height - padding * 3)) - 8, 16, 16);
				
				if(j != 0){
					graphics.setColor(g.getColor());
					graphics.drawLine(
							padding * 2 + (int)((prev.getX() - xMin) / (xMax - xMin) * (width - padding * 3)),
							(height - padding * 2) - (int)((prev.getY() - yMin) / (yMax - yMin) * (height - padding * 3)),
							padding * 2 + (int)((p.getX() - xMin) / (xMax - xMin) * (width - padding * 3)),
							(height - padding * 2) - (int)((p.getY() - yMin) / (yMax - yMin) * (height - padding * 3)));
				}
				prev = p;
			}
		}
	}
	
	public void saveToFile(String path, String type){
		try{
			ImageIO.write(graph, type, new File(path));
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
	public void addPoint(double x, double y, Color c){
		points.add(new Point(x, y, c));
	}
	
	public void addPoint(double x, double y){
		points.add(new Point(x, y));
	}
	
	public void addLine(double m, double b, Color c){
		lines.add(new Line(m, b, c));
	}
	
	public void addLine(double m, double b){
		lines.add(new Line(m, b));
	}
	
	public void addLineGraph(double[] xs, double[] ys){
		ArrayList<Point> arr = new ArrayList<>();
		for(int i = 0; i < xs.length; i++){
			arr.add(new Point(xs[i], ys[i]));
		}
		lineGraphs.add(new LineGraph(arr));
	}
	
	public void addLineGraph(double[] xs, double[] ys, Color c){
		ArrayList<Point> arr = new ArrayList<>();
		for(int i = 0; i < xs.length; i++){
			arr.add(new Point(xs[i], ys[i]));
		}
		lineGraphs.add(new LineGraph(arr, c));
	}
	
	public void addLineGraph(double[] xs, double[] ys, Color[] cs, Color c){
		ArrayList<Point> arr = new ArrayList<>();
		for(int i = 0; i < xs.length; i++){
			arr.add(new Point(xs[i], ys[i], cs[i]));
		}
		lineGraphs.add(new LineGraph(arr, c));
	}
	
	public BufferedImage getGraph(){
		return graph;
	}
	
	public int getWidth(){
		return width;
	}
	
	public int getHeight(){
		return height;
	}
	
	public void dispose(){
		graphics.dispose();
	}
	
	public interface ColorFunction{
		public Color getColor(double x, double y);
	}
}
