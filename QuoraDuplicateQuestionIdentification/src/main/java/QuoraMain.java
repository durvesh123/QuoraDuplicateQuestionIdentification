import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.*;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import scala.Serializable;
import scala.Tuple2;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;


public class QuoraMain implements Serializable {



    public static void main(String args[]) {

        String inputFilePath;
        SparkConf conf;
        SparkContext sc;
        SparkSession spark;
        inputFilePath = "/Users/durveshvedak/IdeaProjects/QuoraDuplicateQuestionIdentification/src/main/resources/train.csv";
        conf = new SparkConf().setAppName("Quora").setMaster("local");
        sc = new SparkContext(conf);
        spark = SparkSession.builder().appName("Quora").getOrCreate();
        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.StringType, false, Metadata.empty()),
                new StructField("qid1", DataTypes.StringType, false, Metadata.empty()),
                new StructField("qid2", DataTypes.StringType, false, Metadata.empty()),
                new StructField("question1", DataTypes.StringType, false, Metadata.empty()),
                new StructField("question2", DataTypes.StringType, false, Metadata.empty()),
                new StructField("is_duplicate", DataTypes.StringType, false, Metadata.empty())
        });

        Dataset<Row> df = spark.read()
                .option("mode", "DROPMALFORMED")
                .option("header", "true")
                .schema(schema)
                .csv(inputFilePath);

        StringIndexer q1indexer = new StringIndexer()
                .setInputCol("question1")
                .setOutputCol("question1Index");

        Dataset<Row> q1indexed = q1indexer.fit(df).transform(df);

        StringIndexer q2indexer = new StringIndexer()
                .setInputCol("question2")
                .setOutputCol("question2Index");

        Dataset<Row> q2indexed = q2indexer.fit(q1indexed.na().drop()).transform(q1indexed.na().drop()).drop("question1");

        Dataset<Row> final_df =  q2indexed.drop("question2");

        Dataset<Row> new_df = final_df.select("is_duplicate", "question1Index", "question2Index");
        //new_df.show();

        JavaRDD<Row> a = new_df.toJavaRDD();
        JavaRDD<String> b = a.map(line->line.toString());
        JavaRDD<String> b1 = b.map(x -> x.replace("[", ""));
        JavaRDD<String> b2 = b1.map(x -> x.replace("]", ""));
        JavaRDD<String>[] data = b2.randomSplit(new double[]{0.7, 0.3},17);

        JavaRDD<String> trainData = data[0];
        JavaRDD<String> testData = data[1];
        trainAndEvaluate(trainData,testData);
    }

    public static void trainAndEvaluate(JavaRDD<String> trainData, JavaRDD<String> testData) {


        //convert input to RDD label points
        JavaRDD<LabeledPoint> training = trainData
                .map(new Function<String, LabeledPoint>() {
                    public LabeledPoint call(String line) throws Exception {
                        String[] parts = line.split(",");
                        return new LabeledPoint(Double.parseDouble(parts[0]),
                                Vectors.dense(Double.parseDouble(parts[1]),
                                        Double.parseDouble(parts[2])));
                    }
                });

        JavaRDD<LabeledPoint> test = testData
                .map(new Function<String, LabeledPoint>() {
                    public LabeledPoint call(String line) throws Exception {
                        String[] parts = line.split(",");
                        return new LabeledPoint(Double.parseDouble(parts[0]),
                                Vectors.dense(Double.parseDouble(parts[1]),
                                        Double.parseDouble(parts[2])));
                    }
                });

        // Run training algorithm to build the model.
        LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
                .run(training.rdd());

        // Clear the prediction threshold so the model will return probabilities
        model.clearThreshold();

        // Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = test.map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(LabeledPoint p) {
                        Double prediction = (double)Math.round(model.predict(p.features()));
                        return new Tuple2<Object, Object>(prediction, p.label());
                    }
                }
        );
        predictionAndLabels.saveAsTextFile("./Output/logisiticRegression/");

        // Get evaluation metrics.
        BinaryClassificationMetrics metrics =
                new BinaryClassificationMetrics(predictionAndLabels.rdd());

        // Get evaluation metrics.
        MulticlassMetrics metrics2 = new MulticlassMetrics(predictionAndLabels.rdd());
        // Accuracy
        System.out.println("Accuracy = " + metrics2.accuracy());

        // Precision by threshold
        JavaRDD<Tuple2<Object, Object>> precision = metrics.precisionByThreshold().toJavaRDD();
        System.out.println("Precision by threshold: " + precision.collect());

        // Recall by threshold
        JavaRDD<?> recall = metrics.recallByThreshold().toJavaRDD();
        System.out.println("Recall by threshold: " + recall.collect());

        // F Score by threshold
        JavaRDD<?> f1Score = metrics.fMeasureByThreshold().toJavaRDD();
        System.out.println("F1 Score by threshold: " + f1Score.collect());

        JavaRDD<?> f2Score = metrics.fMeasureByThreshold(2.0).toJavaRDD();
        System.out.println("F2 Score by threshold: " + f2Score.collect());

        // Precision-recall curve
        JavaRDD<?> prc = metrics.pr().toJavaRDD();
        System.out.println("Precision-recall curve: " + prc.collect());

        // Thresholds
        JavaRDD<Double> thresholds = precision.map(t -> Double.parseDouble(t._1().toString()));

        // ROC Curve
        JavaRDD<?> roc = metrics.roc().toJavaRDD();
        System.out.println("ROC curve: " + roc.collect());

        // AUPRC
        System.out.println("Area under precision-recall curve = " + metrics.areaUnderPR());

        // AUROC
        System.out.println("Area under ROC = " + metrics.areaUnderROC());






        // Save and load model

        //model.save(sc, "target/tmp/RidgeLogisticRegressionModel");
        //LogisticRegressionModel.load(sc, "target/tmp/RidgeLogisticRegressionModel");








    }

}


