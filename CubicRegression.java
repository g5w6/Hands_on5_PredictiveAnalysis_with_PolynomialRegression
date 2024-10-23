import java.util.List;

public class CubicRegression {
    private double B0, B1, B2, B3; // Coeficientes del modelo cúbico
    private final List<double[]> data; // Datos de entrenamiento

    // Constructor que inicializa los datos de entrenamiento
    public CubicRegression(List<double[]> data) {
        this.data = data;
    }

    // Método para entrenar el modelo cúbico usando mínimos cuadrados
    public void trainModel() {
        int n = data.size(); // Número de puntos de datos
        double[] X = new double[n]; // Array para las variables independientes (Batch Size)
        double[] Y = new double[n]; // Array para las variables dependientes (outputs)

        // Rellenar los arrays X e Y con los datos
        for (int i = 0; i < n; i++) {
            X[i] = data.get(i)[0];
            Y[i] = data.get(i)[1];
        }

        // Crear la matriz X para la regresión cúbica
        double[][] X_matrix = new double[n][4];
        for (int i = 0; i < n; i++) {
            X_matrix[i][0] = 1; // Término independiente
            X_matrix[i][1] = X[i]; // Término lineal
            X_matrix[i][2] = Math.pow(X[i], 2); // Término cuadrático
            X_matrix[i][3] = Math.pow(X[i], 3); // Término cúbico
        }

        // Transponer la matriz X_matrix
        double[][] X_transpose = new double[4][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 4; j++) {
                X_transpose[j][i] = X_matrix[i][j];
            }
        }

        // Calcular el producto (X^T * X)
        double[][] XTX = new double[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < n; k++) {
                    XTX[i][j] += X_transpose[i][k] * X_matrix[k][j];
                }
            }
        }

        // Calcular el producto (X^T * Y)
        double[] XTY = new double[4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < n; j++) {
                XTY[i] += X_transpose[i][j] * Y[j];
            }
        }

        // Resolver el sistema de ecuaciones (XTX * B = XTY) para obtener los coeficientes B
        double[] B = gaussianElimination(XTX, XTY);
        B0 = B[0];
        B1 = B[1];
        B2 = B[2];
        B3 = B[3];
    }

    // Método para realizar la eliminación gaussiana y resolver el sistema lineal
    private double[] gaussianElimination(double[][] A, double[] B) {
        int n = B.length;
        for (int p = 0; p < n; p++) {
            // Encontrar la fila pivote y hacer el intercambio
            int max = p;
            for (int i = p + 1; i < n; i++) {
                if (Math.abs(A[i][p]) > Math.abs(A[max][p])) {
                    max = i;
                }
            }
            double[] temp = A[p];
            A[p] = A[max];
            A[max] = temp;
            double t = B[p];
            B[p] = B[max];
            B[max] = t;

            // Pivotear dentro de A y B
            for (int i = p + 1; i < n; i++) {
                double alpha = A[i][p] / A[p][p];
                B[i] -= alpha * B[p];
                for (int j = p; j < n; j++) {
                    A[i][j] -= alpha * A[p][j];
                }
            }
        }

        // Sustitución hacia atrás para encontrar la solución
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int j = i + 1; j < n; j++) {
                sum += A[i][j] * x[j];
            }
            x[i] = (B[i] - sum) / A[i][i];
        }
        return x;
    }

    // Método para predecir el valor de Y dado un Batch Size usando el modelo cúbico
    public double predict(double batchSize) {
        return B0 + B1 * batchSize + B2 * Math.pow(batchSize, 2) + B3 * Math.pow(batchSize, 3);
    }

    // Calcular el Error Cuadrático Medio (MSE)
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

    // Calcular la correlación de Pearson
    public double correlation() {
        double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0, sumYY = 0;
        int n = data.size();

        for (double[] entry : data) {
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

    // Calcular el coeficiente de determinación (R-squared)
    public double RSquared(List<double[]> testData) {
        double meanY = 0;
        for (double[] entry : testData) {
            meanY += entry[1];
        }
        meanY /= testData.size();

        double sumatotalcuadrados = 0, residualsumacuadrados = 0;
        for (double[] entry : testData) {
            double actualY = entry[1];
            double predictedY = predict(entry[0]);
            sumatotalcuadrados += Math.pow(actualY - meanY, 2);
            residualsumacuadrados += Math.pow(actualY - predictedY, 2);
        }
        return 1 - (residualsumacuadrados / sumatotalcuadrados);
    }

    // Métodos getter para obtener los coeficientes del modelo
    public double getB0() {
        return B0;
    }

    public double getB1() {
        return B1;
    }

    public double getB2() {
        return B2;
    }

    public double getB3() {
        return B3;
    }
}