
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra.Double;

namespace SharpLearn.DataProcessors {

    public class DataGenerator {


        public Matrix<double> Generate2DLineData(double slope, double bias, int size, double noiseStdDev) {
            Random random = new Random();
            Normal normalDist = new Normal(0, noiseStdDev) { RandomSource = random };

            // Generate X values uniformly between -10 and 10
            Vector<double> xValues = Vector<double>.Build.Random(size, new ContinuousUniform(-10, 10, random));

            // Generate noise values
            Vector<double> noise = Vector<double>.Build.Random(size, normalDist);

            // Calculate Y values as y = m * x + b + noise
            Vector<double> yValues = xValues.Map(x => slope * x + bias) + noise;

            // Combine X and Y into a single matrix
            return Matrix<double>.Build.DenseOfColumnVectors(xValues, yValues);
        }
    }
}