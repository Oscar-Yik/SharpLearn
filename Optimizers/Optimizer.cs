
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System.Collections.Generic;
using SharpLearn.LossFunctions;

namespace SharpLearn.Optimizers {

    public abstract class Optimizer {

        protected double _learningRate; 
        protected Vector<double>? _parameters; 
        protected ILossFunction? _lossFunction;
        protected LossFunctionArguments? _lossFunctionArguments; 

        public Vector<double>? Parameters {
            get => _parameters;
            set => _parameters = value;
        }

        public abstract StepResult Step(); 

        public abstract void Reset();

        public abstract void Clear();

        public void SetLearningRate(double learningRate) {
            _learningRate = learningRate;
        }

        public void SetParameters(Vector<double> parameters) {
            Parameters = parameters;
        }

        public void SetLossFunction(ILossFunction lossFunction) {
            _lossFunction = lossFunction;
        }

        public void SetLossFunctionArguments(LossFunctionArguments lossFunctionArguments) {
            _lossFunctionArguments = lossFunctionArguments;
        }
    }

    public class StepResult {
        public double FNew { get; }
        public Vector<double>? GNew { get; } 
        public Vector<double>? WNew { get; }
        public bool BreakYes { get; }

        public StepResult(double fNew, Vector<double> gNew, Vector<double> wNew, bool breakYes) {
            FNew = fNew; 
            GNew = gNew; 
            WNew = wNew;
            BreakYes = breakYes;
        }
    }
}