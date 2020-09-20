using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace myMLAPI
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World! This is My First ML API :)");
            //
            MLContext mlContext = new MLContext();

            IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>
                (
                path: "C:\\Users\\PC\\source\\repos\\myMLAPI\\myMLAPI\\wikipedia-detox-250-line-data.tsv",
                hasHeader:true,
                separatorChar: '\t',
                allowQuoting:true,
                allowSparse:false);

            IEstimator<ITransformer> pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(ModelInput.SentimentText))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Sentiment",featureColumnName: "Features"));

            ITransformer model = pipeline.Fit(dataView);
            var metrics = mlContext.BinaryClassification.CrossValidateNonCalibrated(dataView, pipeline,numberOfFolds:5,labelColumnName: "Sentiment");
            PredictionEngine<ModelInput, ModelOutput> prediction = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

            var Sample = new ModelInput() { SentimentText = "Excellent" };
            var result = prediction.Predict(Sample);
            Console.WriteLine("Toxic Statment: {0}",result.Prediction);
        }
    }
    public class ModelInput
    {
        [ColumnName("Sentiment"),LoadColumn(0)]
        public bool Sentiment { get; set; }

        [ColumnName("SentimentText"), LoadColumn(1)]
        public string SentimentText { get; set; }
    }
    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

    }
}
