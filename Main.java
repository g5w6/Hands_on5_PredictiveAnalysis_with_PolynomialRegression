import java.util.List;

public class Main {
    public static void main(String[] args) {
        DataSet dataset = new DataSet();

        // Modelo usando el 100% de los datos
        System.out.println("Prueba con 100% del dataset: ");
        SimpleLinearRegression model100 = new SimpleLinearRegression(dataset.getAllData());
        model100.trainModel();
        printModelDetails("Regresion Lineal Simple", model100, dataset.getAllData());

        CubicRegression cubicModel100 = new CubicRegression(dataset.getAllData());
        cubicModel100.trainModel();
        printCubicModelDetails("Regresion Cubica", cubicModel100, dataset.getAllData());

        // Modelo con 70% entrenamiento y 30% prueba
        System.out.println("\nPrueba con 70% entrenamiento y 30% prueba:");
        List<double[]> trainingSet70 = dataset.getTrainingSet(0.7);
        List<double[]> testSet30 = dataset.getTestSet(0.3);
        SimpleLinearRegression model70_30 = new SimpleLinearRegression(trainingSet70);
        model70_30.trainModel();
        printModelDetails("Regresion Lineal Simple", model70_30, testSet30);

        CubicRegression cubicModel70_30 = new CubicRegression(trainingSet70);
        cubicModel70_30.trainModel();
        printCubicModelDetails("Regresion Cubica", cubicModel70_30, testSet30);

        // Modelo con 30% entrenamiento y 70% prueba
        System.out.println("\nPrueba con 30% entrenamiento y 70% prueba:");
        List<double[]> trainingSet30 = dataset.getTrainingSet(0.3);
        List<double[]> testSet70 = dataset.getTestSet(0.7);
        SimpleLinearRegression model30_70 = new SimpleLinearRegression(trainingSet30);
        model30_70.trainModel();
        printModelDetails("Regresion Lineal Simple", model30_70, testSet70);

        CubicRegression cubicModel30_70 = new CubicRegression(trainingSet30);
        cubicModel30_70.trainModel();
        printCubicModelDetails("Regresion Cubica", cubicModel30_70, testSet70);
    }

    // Método para imprimir los detalles del modelo (B0, B1, MSE, correlación, predicciones)
    public static void printModelDetails(String modelName, SimpleLinearRegression model, List<double[]> testSet) {
        System.out.println(modelName + " Ecuacion de la recta: y = " + model.getB1() + " + " + model.getB0());
        System.out.println("MSE: " + model.MSE(testSet));
        System.out.println("Correlacion: " + model.correlation());
        System.out.println("R-squared: " + model.RSquared(testSet));

        // Realizar y mostrar predicciones
        double[] batchSizesToPredict = {50, 100, 150, 30, 70};
        for (double batchSize : batchSizesToPredict) {
            System.out.println(modelName + " Prediccion para Batch Size " + batchSize + ": " + model.predict(batchSize));
        }
    }

    // Método para imprimir los detalles del modelo cúbico (B0, B1, B2, B3, MSE, correlación, predicciones)
    public static void printCubicModelDetails(String modelName, CubicRegression model, List<double[]> testSet) {
        System.out.println(modelName + " Ecuacion de la curva: y = " + model.getB3() + "x^3 + " + model.getB2() + "x^2 + " + model.getB1() + "x + " + model.getB0());
        System.out.println("MSE: " + model.MSE(testSet));
        System.out.println("Correlacion: " + model.correlation());
        System.out.println("R-squared: " + model.RSquared(testSet));

        // Realizar y mostrar predicciones
        double[] batchSizesToPredict = {50, 100, 150, 30, 70};
        for (double batchSize : batchSizesToPredict) {
            System.out.println(modelName + " Prediccion para Batch Size " + batchSize + ": " + model.predict(batchSize));
        }
    }
}