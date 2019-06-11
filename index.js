let net;
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();


async function setupWebcam() {
  return new Promise((resolve, reject) => {
    const navigatorAny = navigator;
    navigator.getUserMedia = navigator.getUserMedia ||
        navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
        navigatorAny.msGetUserMedia;
    if (navigator.getUserMedia) {
      navigator.getUserMedia({video: true},
        stream => {
          webcamElement.srcObject = stream;
          webcamElement.addEventListener('loadeddata',  () => resolve(), false);
        },
        error => reject());
    } else {
      reject();
    }
  });
}

const saveData = () => {
  let dataset = classifier.getClassifierDataset();
  var datasetObj = {};
  Object.keys(dataset).forEach((key) => {
    let data = dataset[key].dataSync();
    console.log(dataset[key].shape)
    // use Array.from() so when JSON.stringify() it covert to an array string e.g [0.1,-0.2...] 
    // instead of object e.g {0:"0.1", 1:"-0.2"...}
    datasetObj[key] = Array.from(data); 
  });
  let jsonStr = JSON.stringify(datasetObj);

  // console.log(jsonStr);
  localStorage.setItem("mlData", jsonStr);
}

const loadData = () => {
  let dataset = localStorage.getItem("mlData")
  let tensorObj = JSON.parse(dataset)
  //covert back to tensor
  Object.keys(tensorObj).forEach((key) => {
    tensorObj[key] = tf.tensor(tensorObj[key], [tensorObj[key].length / 1024, 1024])
  })
  classifier.setClassifierDataset(tensorObj);
}

async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();
  console.log('Sucessfully loaded model');
  
  await setupWebcam();

  const addExample = classId => {
    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    const activation = net.infer(webcamElement, 'conv_preds');

    console.log(`${classId} => ${activation}`)

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);
  };

  document.getElementById('class-a').addEventListener('click', () => addExample(0));
  document.getElementById('class-b').addEventListener('click', () => addExample(1));
  document.getElementById('class-c').addEventListener('click', () => addExample(2));
  document.getElementById('save-data').addEventListener('click', () => saveData());
  document.getElementById('load-data').addEventListener('click', () => loadData());

  while (true) {
    if (classifier.getNumClasses() > 0) {
      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(webcamElement, 'conv_preds');
      // Get the most likely class and confidences from the classifier module.
      const result = await classifier.predictClass(activation);

      const classes = ['A', 'B', 'C'];
      document.getElementById('console').innerText = `
        prediction: ${classes[result.classIndex]}\n
        probability: ${result.confidences[result.classIndex]}
      `;

      classifier.getClassifierDataset();
    }

    // Give some breathing room by waiting for the next animation frame to
    // fire.
    await tf.nextFrame();
  }
}

app();