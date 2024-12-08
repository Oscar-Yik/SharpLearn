using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System.Collections.Generic;
using SharpLearn.LossFunctions;
using System;

namespace SharpLearn.Optimizers {

    public class GradientDescent : Optimizer {
        private double _optimalTolerance; 
        private double _initialLearningRate; 
        private int _maxEvals; 
        private int _numEvals; 
        private double? _fOld; // can be null
        private Vector<double>? _gOld; // can be null

        public GradientDescent(double optimalTolerance, double learningRate, int maxEvals) {
            Parameters = null; 
            _optimalTolerance = optimalTolerance;
            _learningRate = learningRate;
            _initialLearningRate = _learningRate;
            _maxEvals = maxEvals;
            _numEvals = 0;

            _fOld = null;
            _gOld = null;
        }

        public override StepResult Step() {
            if (_fOld == null || _gOld == null) {
                EvaluateResult evaluateResult = GetFunctionValueAndGradient(Parameters);
                _fOld = evaluateResult.F;
                _gOld = evaluateResult.G; 
            }

            GetLearningRateAndStepResults res = GetLearningRateAndStep(_gOld);
            Parameters = res.WNew; 

            if (_numEvals < 4) {
                Console.WriteLine($"Stepping number {_numEvals}, w: {res.WNew[0]}, g: {res.GNew}, f: {res.FNew}");
            }

            _fOld = res.FNew; 
            _gOld = res.GNew; 

            _numEvals++;
            bool breakYes = BreakYes(res.GNew);
            return new StepResult(res.FNew, res.GNew, Parameters, breakYes);
        }

        private GetLearningRateAndStepResults GetLearningRateAndStep(Vector<double> gOld) {
            Vector<double> wOld = Parameters; 
            if (wOld == null) {
                throw new ArgumentNullException("wOld is null");
            }
            if (wOld.Count == 0) {
                throw new ArgumentNullException("wOld is empty");
            }
            double alpha = _learningRate;
            Vector<double> wNew = wOld - alpha * gOld; 
            // if (wNew == null) {
            //     throw new ArgumentNullException("wNew is null");
            // }
            // if (wNew.Count == 0) {
            //     throw new ArgumentNullException("wNew is empty");
            // }
            EvaluateResult evaluateResult = GetFunctionValueAndGradient(wNew);
            return new GetLearningRateAndStepResults(wNew, evaluateResult);
        }

        private bool BreakYes(Vector<double> g) {
            // infinity norm of the gradient 
            // if (g == null) {
            //     throw new ArgumentNullException("g is null");
            // }
            // if (g.Count == 0) {
            //     throw new ArgumentNullException("g is empty");
            // }
            double gradient_norm = g.PointwiseAbs().Maximum(); 
            if (gradient_norm < _optimalTolerance) {
                return true;
            } else if (_numEvals >= _maxEvals) {
                return true; 
            } else {
                return false; 
            }
        }

        private Vector<double> GetNextParameterValue(double alpha, Vector<double> g) {
            return Parameters - alpha * g; 
        }

        private EvaluateResult GetFunctionValueAndGradient(Vector<double> w) {
            return _lossFunction.Evaluate(w, _lossFunctionArguments);
        }

        public override void Reset() {
            _numEvals = 0;
            Parameters = null;
            _lossFunctionArguments = null;
            _learningRate = _initialLearningRate;
            _fOld = null;
            _gOld = null; 
        }

        public override void Clear() {
            _fOld = null;
            _gOld = null; 
        }
    }

    public class GetLearningRateAndStepResults {
        public Vector<double>? WNew { get; }
        public double FNew { get; }
        public Vector<double>? GNew { get; } 

        public GetLearningRateAndStepResults(Vector<double> wNew, EvaluateResult evaluateResult) {
            WNew = wNew;
            FNew = evaluateResult.F; 
            GNew = evaluateResult.G; 
        }
    }
}