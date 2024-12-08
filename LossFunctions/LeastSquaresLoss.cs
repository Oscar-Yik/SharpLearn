
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace SharpLearn.LossFunctions {

    public class LeastSquaresLoss : ILossFunction {

        public EvaluateResult Evaluate(Vector<double> w, LossFunctionArguments lossFunctionArguments) {
            Matrix<double> X = lossFunctionArguments.X; 
            Vector<double> y = lossFunctionArguments.Y; 

            Vector<double> yHat = X * w; 
            Vector<double> mResiduals = yHat - y; 
            double f = 0.5 * mResiduals.PointwisePower(2).Sum();
            Vector<double> g = X.TransposeThisAndMultiply(mResiduals); 
            return new EvaluateResult(f, g); 
        }
    }

    public class EvaluateResult {
        public double F { get; }
        public Vector<double>? G { get; } 

        public EvaluateResult(double f, Vector<double> g) {
            F = f;
            G = g; 
        }
    }

    public class LossFunctionArguments {
        public Matrix<double>? X { get; } 
        public Vector<double>? Y { get; }

        public LossFunctionArguments(Matrix<double> x, Vector<double> y) {
            X = x;
            Y = y; 
        }
    }
}