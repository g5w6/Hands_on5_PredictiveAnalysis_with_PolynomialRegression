import java.util.List;

public class SimpleLinearRegression {
    private double B0, B1;
    private final List<double[]> data;

    public SimpleLinearRegression(List<double[]> data) {
        this.data = data;
    }

    public void trainModel(){
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        int n = data.size();

        for (double[] entry : data){
            double x = entry[0];
            double y = entry[1];
            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumX2 += x * x;
        }

        B1 = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        B0 = (sumY - B1 * sumX) / n;
    }

    public double predict(double batchSize){
        return B0 + B1 * batchSize;
    }

    // Caluclar el MSE
    public double MSE(List<double[]> testSet) {
        double mse = 0;
        for (double[] row : testSet) {
            double x = row[0];
            double y = row[1];
            double predictedY = predict(x);
            mse += Math.pow(y - predictedY, 2);
        }
        return mse / testSet.size();
    }

    // Calcular la correlacion de Pearson
    public double correlation(){
        double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0, sumYY = 0;
        int n = data.size();

        for(double[] entry : data){
            double x = entry[0];
            double y = entry[1];
            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumXX += x * x;
            sumYY += y * y;
        }
        double numerator = n * sumXY - sumX * sumY;
        double denominator = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
        return numerator / denominator;
    }

    // Calcular R-squared -- Coeficiente de Determinación
    public double RSquared(List<double[]> testData){
        double meanY = 0;
        for(double[] entry : testData){
            meanY += entry[1];
        }
        meanY /= testData.size();

        double sumatotalcuadrados = 0, residualsumacuadrados = 0;
        for(double[] entry : testData){
            double actualY = entry[1];
            double predictedY = predict(entry[0]);
            sumatotalcuadrados += Math.pow(actualY - meanY, 2);
            residualsumacuadrados += Math.pow(actualY - predictedY, 2);
        }
        return 1 - (residualsumacuadrados / sumatotalcuadrados);
    }
    public double getB0(){
        return B0;
    }
    public double getB1(){
        return B1;
    }
}
