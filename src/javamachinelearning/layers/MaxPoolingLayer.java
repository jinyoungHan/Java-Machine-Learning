package javamachinelearning.layers;

import javamachinelearning.regularizers.Regularizer;
import javamachinelearning.utils.Tensor;

public class MaxPoolingLayer implements Layer{
	private int[] prevShape;
	private int[] nextShape;
	private int winWidth, winHeight;
	private int strideX, strideY;
	private int[][] maxIdx;
	
	public MaxPoolingLayer(int winWidth, int winHeight, int strideX, int strideY){
		this.winWidth = winWidth;
		this.winHeight = winHeight;
		this.strideX = strideX;
		this.strideY = strideY;
	}
	
	public MaxPoolingLayer(int winSize, int stride){
		this.winWidth = winSize;
		this.winHeight = winSize;
		this.strideX = stride;
		this.strideY = stride;
	}
	
	public MaxPoolingLayer(int winSize){
		this.winWidth = winSize;
		this.winHeight = winSize;
		this.strideX = 1;
		this.strideY = 1;
	}
	
	@Override
	public int[] nextShape(){
		return nextShape;
	}
	
	@Override
	public int[] prevShape(){
		return prevShape;
	}
	
	@Override
	public void init(int[] prevShape){
		this.prevShape = prevShape;
		
		int temp = prevShape[0] - winWidth;
		if(temp % strideX != 0)
			throw new IllegalArgumentException("Bad sizes for max pooling!");
		int w = temp / strideX + 1;
		
		temp = prevShape[1] - winHeight;
		if(temp % strideY != 0)
			throw new IllegalArgumentException("Bad sizes for max pooling!");
		int h = temp / strideY + 1;
		
		nextShape = new int[]{w, h, prevShape[2]};
		maxIdx = new int[nextShape[0] * nextShape[1] * nextShape[2]][2];
	}
	
	@Override
	public Tensor forwardPropagate(Tensor input, boolean training){
		double[] res = new double[nextShape[0] * nextShape[1] * nextShape[2]];
		int[] shape = input.shape();
		int idx = 0;
		// slide through and computes the max for each location
		// the output should have the same depth as the input
		for(int i = 0; i < nextShape[0] * strideX; i += strideX){
			for(int j = 0; j < nextShape[1] * strideY; j += strideY){
				for(int k = 0; k < shape[2]; k++){ // for each depth slice
					double max = Double.MIN_VALUE;
					
					for(int rx = 0; rx < winWidth; rx++){ // relative x position
						for(int ry = 0; ry < winHeight; ry++){ // relative y position
							// absolute positions
							int x = i + rx;
							int y = j + ry;
							double val = input.flatGet(x * shape[1] * shape[2] + y * shape[2] + k);
							
							if(val > max){
								max = val;
								
								maxIdx[idx][0] = x;
								maxIdx[idx][1] = y;
							}
						}
					}
					
					// max of all values
					res[idx] = max;
					idx++;
				}
			}
		}
		
		return new Tensor(nextShape, res);
	}
	
	@Override
	public Tensor backPropagate(Tensor prevRes, Tensor nextRes, Tensor error, Regularizer regularizer){
		double[] res = new double[prevShape[0] * prevShape[1] * prevShape[2]];
		int outIdx = 0;
		
		for(int i = 0; i < nextShape[0] * strideX; i += strideX){
			for(int j = 0; j < nextShape[1] * strideY; j += strideY){
				for(int k = 0; k < prevShape[2]; k++){ // for each depth slice
					for(int rx = 0; rx < winWidth; rx++){ // relative x position
						for(int ry = 0; ry < winHeight; ry++){ // relative y position
							// absolute positions
							int x = i + rx;
							int y = j + ry;
							int inIdx = x * prevShape[1] * prevShape[2] + y * prevShape[2] + k;
							
							if(maxIdx[outIdx][0] == x && maxIdx[outIdx][1] == y){
								res[inIdx] += error.flatGet(outIdx);
							}
						}
					}
					
					outIdx++;
				}
			}
		}
		
		return new Tensor(prevShape, res);
	}
}