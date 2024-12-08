
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace SharpLearn.LossFunctions {

    public interface ILossFunction {
        EvaluateResult Evaluate(Vector<double> w, LossFunctionArguments lossFunctionArguments);
    }

}