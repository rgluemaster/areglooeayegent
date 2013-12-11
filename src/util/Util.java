package util;

public class Util {

	public static double norm(double[] vector) {
		double sum = 0;
		for(int i = 0;i<vector.length;i++) {
			sum += vector[i]*vector[i];
		}
		return Math.sqrt(sum);
	}
	
	public static double roundNDecimals(double d, int N) {
		double factor = Math.pow(10, N);
		double result = d * factor;
		result = Math.round(result);
		result = result / factor;
		return result;
	}
	
	public static double arraySum(double[] array) {
		double sum = 0;
		for(int i = 0;i<array.length;i++) {
			sum += array[i];
		}
		return sum;
	}

}
