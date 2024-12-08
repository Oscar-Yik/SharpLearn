using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System.Collections.Generic;
using SharpLearn.Optimizers; 
using SharpLearn.LossFunctions;
using System;

namespace SharpLearn.LinearModels {

    public class LinearModel {
        
        private ILossFunction _lossFunction; 
        private Optimizer _optimizer; 
        // private bool _hasBias; 
        private List<double> _fs; 
        private List<Vector<double>> _ws; 
        private List<Vector<double>>  _gs; 
        private Vector<double>? _w; 

        public LinearModel(ILossFunction lossFunction, Optimizer optimizer) {
            _lossFunction = lossFunction;
            _optimizer = optimizer;
            // _hasBias = true; 
            _fs = new List<double>();
            _ws = new List<Vector<double>>();
            _gs = new List<Vector<double>>();
        }

        public virtual OptimizeResult Optimize(Vector<double> wInit, Matrix<double> X, Vector<double> y) {
            Vector<double> w = wInit.Clone(); 
            LossFunctionArguments args = new LossFunctionArguments(X, y);
            EvaluateResult res = _lossFunction.Evaluate(w, args);
            // Console.WriteLine(wInit);

            _optimizer.Reset();
            _optimizer.SetLossFunction(_lossFunction);
            _optimizer.SetParameters(w);
            _optimizer.SetLossFunctionArguments(args);

            _fs.Add(res.F);
            _gs.Add(res.G);
            
            while (true) {
                // Console.WriteLine("Stepping");
                StepResult result = _optimizer.Step(); 
                w = result.WNew;
                _fs.Add(result.FNew);
                _gs.Add(result.GNew);
                _ws.Add(result.WNew);
                if (result.BreakYes) {
                    break; 
                }
            }

            return new OptimizeResult(w, _fs, _ws, _gs);
        }

        public virtual void Fit(Matrix<double> X, Vector<double> y) {
            int n = X.RowCount; 
            int d = X.ColumnCount; 

            Vector<double> w = Vector<double>.Build.Dense(d, 0.0);
            Console.WriteLine(d);

            OptimizeResult result = Optimize(w, X, y);
            _w = result.W;
            _fs = result.Fs; 
            _gs = result.Gs; 
            _ws = result.Ws; 
        }

        public virtual Vector<double> Predict(Matrix<double> X) {
            return X * _w; 
        }

    }

    public class OptimizeResult {
        public Vector<double>? W { get; } 
        public List<double>? Fs { get; }
        public List<Vector<double>>? Ws { get; } 
        public List<Vector<double>>? Gs { get; } 

        public OptimizeResult(Vector<double> w, List<double> fs, List<Vector<double>> ws, List<Vector<double>> gs) {
            W = w;
            Fs = fs;
            Ws = ws;
            Gs = gs; 
        }
    }
}