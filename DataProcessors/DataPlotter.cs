
using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using ScottPlot;

namespace SharpLearn.DataProcessors {

    public class DataPlotter {

        public DataPlotter() {
            
        }

        // matrix must only have 2 rows
        public (double[], double[]) ConvertMatrixTo2DArray(Matrix<double> matrix) {
            return (matrix.Row(0).ToArray(), matrix.Row(1).ToArray()); 
        }

        // matrix must only have 2 rows 
        public void Plot2DscatterPlot(Matrix<double> original, Matrix<double> prediction) {
            Plot plt = new();

            (double[] x, double[] y) = ConvertMatrixTo2DArray(original); 
            (double[] xHat, double[] yHat) = ConvertMatrixTo2DArray(prediction); 
            var scatter = plt.Add.ScatterPoints(x, y);
            scatter.Color = Colors.Blue; 

            var line = plt.Add.Scatter(xHat, yHat);
            line.Color = Colors.Red; 

            plt.Title("Least Squares Gradient Descent");
            plt.XLabel("X-Axis");
            plt.YLabel("Y-Axis");

            plt.SavePng("sharpLearn_firstModel.png", 600, 400);
            Console.WriteLine($"Scatter plot saved to sharpLearn_firstModel.png");
        }

    }
}