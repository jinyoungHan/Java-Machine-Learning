package javamachinelearning.optimizers;

import javamachinelearning.utils.Tensor;

public class AdadeltaOptimizer implements Optimizer{
	private static final double epsilon = 0.00000001;
	private double rho;
	private double stepSize; // step size

	public AdadeltaOptimizer(){
		this.rho = 0.9;
		this.stepSize = 0.1;
	}

	public AdadeltaOptimizer(double stepSize){
		this.rho = 0.9;
		this.stepSize = stepSize;
	}
	
	public void updateRate(Tensor t) {
		//stepSize = rho*stepSize - (1-rho)*t;
	}

	@Override
	public int extraParams(){
		return 1;
	}

	@Override
	public void update(){
		// nothing to do
	}

	@Override
	public Tensor optimize(Tensor grads, Tensor[] params){
		// parameter: exponential average of squared gradients
		params[0] = params[0].mul(rho).add((grads.mul(grads)).mul(1.0-rho));
		Tensor t = grads.mul(stepSize).div(params[0].map(x -> Math.sqrt(x)).add(epsilon));
		updateRate(t);
		return t;
	}
}
