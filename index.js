// require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

function knn(features, labels, predictionPoint, k) {
  return (
    features
      .sub(predictionPoint)
      .pow(2)
      .sum(1)
      .pow(0.5)
      .expandDims(1)
      .concat(labels, 1)
      .unstack() //turns it into JS array so we can run sort method
      .sort((a, b) => (a.arraySync()[0] > b.arraySync()[0] ? 1 : -1)) //a.get(0) depr now === a.arraySync()[0]
      .slice(0, k)
      .reduce((acc, pair) => acc + pair.arraySync()[1], 0) / k
  ); //total of top k labels divided by k
}

let { features, labels, testFeatures, testLabels } = loadCSV(
  "kc_house_data.csv",
  {
    shuffle: true,
    splitTest: 10,
    dataColumns: ["lat", "long"],
    labelColumns: ["price"],
  }
);

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testpoint, i) => {
  const result = knn(features, labels, tf.tensor(testpoint), 10);
  const err = ((testLabels[i][0] - result) / testLabels[i][0]) * 100;
  console.log("res:", result, "err:", err);
});
