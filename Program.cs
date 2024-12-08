
using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

using SharpLearn.Optimizers;
using SharpLearn.LossFunctions;
using SharpLearn.LinearModels;
using SharpLearn.DataProcessors;

class Program {
    
    static void Main(string[] args) {
        // Console.WriteLine("Hello, World!");
        // Matrix<double> A = DenseMatrix.OfArray(new double[,] {
        //     {1,1,1,1},
        //     {1,2,3,4},
        //     {4,3,2,1}});
        // Vector<double>[] nullspace = A.Kernel();
        // Console.WriteLine(A * (2*nullspace[0] - 3*nullspace[1]));

        // var A = Matrix<double>.Build.DenseOfArray(new double[,] {
        //     {3,2,-1},
        //     {2,-2,4},
        //     {-1,0.5,-1}});
        // var b = Vector<double>.Build.Dense(new double[] {1, -2, 0 });
        // var x = A.Solve(b);
        // Console.WriteLine($"Vector: [{x[0]:F0}, {x[1]:F0}, {x[2]:F0}]");

        DataGenerator dataGenerator = new DataGenerator(); 
        DataPlotter dataPlotter = new DataPlotter(); 

        Matrix<double> data = dataGenerator.Generate2DLineData(0.5, 0, 100, 5);
        Matrix<double> X = data.SubMatrix(0, data.RowCount, 0, 1);
        Vector<double> y = data.Column(1);

        ILossFunction lossFunc = new LeastSquaresLoss(); 
        Optimizer optimizer = new GradientDescent(0, 0.0001, 20);
        LinearModel model = new LinearModel(lossFunc, optimizer);
        model.Fit(X, y);
        Vector<double> yHat = model.Predict(X);  
        Matrix<double> prediction = X.Transpose().Stack(yHat.ToRowMatrix());

        dataPlotter.Plot2DscatterPlot(data.Transpose(), prediction);
    }
}