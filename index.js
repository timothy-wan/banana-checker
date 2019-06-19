


const walletImg = ["./img/IMG_2267.jpg", "./img/IMG_2275.jpg", "./img/IMG_2276.jpg", "./img/IMG_2277.jpg", "./img/IMG_2278.jpg", "./img/IMG_2279.jpg", "./img/IMG_2280.jpg"];


// Create a KNN classifier
const knnClassifier = ml5.KNNClassifier();

// Create a featureExtractor that can extract features of an image
console.log("model is loading")
const featureExtractor = ml5.featureExtractor("MobileNet", modelReady);



// Add an example with a label to the KNN Classifier
// knnClassifier.addExample(logits, "apple");

// Use KNN Classifier to classify these features
// knnClassifier.classify(features, function(err, result) {
//   console.log(result); // result.label is the predicted label
// });


function modelReady() {
  console.log("model is ready");
}

// Save dataset as myKNNDataset.json
function saveMyKNN() {
  knnClassifier.save('myKNNDataset');
}
// Load dataset to the classifier
function loadMyKNN() {
  knnClassifier.load('https://crossorigin.me/./myKNNDataset.json', () => {
    console.log("model loaded")
  });
}

function trainBanana() {
  const img5 = document.getElementById("img5")
  const logits = featureExtractor.infer( img5);
  knnClassifier.addExample(logits, "banana");

  const img6 = document.getElementById("img6")
  const logits1 = featureExtractor.infer( img6);
  knnClassifier.addExample(logits1, "banana");

  const img7 = document.getElementById("img7")
  const logits2 = featureExtractor.infer( img7);
  knnClassifier.addExample(logits2, "banana");

  const img8 = document.getElementById("img8")
  const logits3 = featureExtractor.infer( img8);
  knnClassifier.addExample(logits3, "banana");
  console.log("trained");

}
function trainApple () {
  

  // for (let i = 0; i < walletImg.length; i++) {
    
    const img4 = document.getElementById("img4")
    const logits = featureExtractor.infer( img4);
    knnClassifier.addExample(logits, "apple");

    const img1 = document.getElementById("img1")
    const logits1 = featureExtractor.infer( img1);
    knnClassifier.addExample(logits1, "apple");

    const img2 = document.getElementById("img2")
    const logits2 = featureExtractor.infer( img2);
    knnClassifier.addExample(logits2, "apple");

    const img3 = document.getElementById("img3")
    const logits3 = featureExtractor.infer( img3);
    knnClassifier.addExample(logits3, "apple");
    console.log("trained");
  // }
  
  
}

function classify() {
  const img = document.getElementById("img-test")
  const logits = featureExtractor.infer( img );
  knnClassifier.classify(logits, (err, results) => {
    if (err) {
      console.error(err);
    }
    console.log(results);
  });
}


document.getElementById("predict").addEventListener("click", classify);
document.getElementById("apple-image-add").addEventListener("click", trainApple);

document.getElementById("banana-image-add").addEventListener("click", trainBanana);
document.getElementById("save-data").addEventListener("click", saveMyKNN);
document.getElementById("load-data").addEventListener("click", loadMyKNN);
document.getElementById("class-a").addEventListener("click", () => {
  console.log("apple");
});