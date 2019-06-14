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
  
  // await setupWebcam();

  const addImage = async (classId) => {
    const walletPhotoArr = ["./img/IMG_2267.jpg", "./img/IMG_2275.jpg", "./img/IMG_2276.jpg", "./img/IMG_2277.jpg", "./img/IMG_2278.jpg", "./img/IMG_2279.jpg", "./img/IMG_2280.jpg"];

    const bananaPhotoArr = ["https://media.istockphoto.com/photos/banana-picture-id636739634?k=6&m=636739634&s=612x612&w=0&h=BQ9Z6DobjFzclh3LN7nKSljrRqycJPCq65CS8rtUHU4=",
    "https://images-na.ssl-images-amazon.com/images/I/71gI-IUNUkL._SY355_.jpg",
  "https://img.purch.com/w/660/aHR0cDovL3d3dy5saXZlc2NpZW5jZS5jb20vaW1hZ2VzL2kvMDAwLzA2NS8xNDkvb3JpZ2luYWwvYmFuYW5hcy5qcGc=",
"https://cdn1.medicalnewstoday.com/content/images/headlines/271/271157/bananas.jpg",
"https://thumbs-prod.si-cdn.com/_oO5E4sOE9Ep-qk_kuJ945_-qo4=/800x600/filters:no_upscale()/https://public-media.si-cdn.com/filer/d5/24/d5243019-e0fc-4b3c-8cdb-48e22f38bff2/istock-183380744.jpg",
"https://www.healthxchange.sg/sites/hexassets/Assets/food-nutrition/good-reasons-to-eat-a-banana-today.jpg"];

    if (classId === "wallet") {

      for (let i = 0; i < walletPhotoArr.length; i++) { 
        var img = new Image(250, 250); // Use DOM HTMLImageElement
        img.src =  walletPhotoArr[i];
        img.alt = 'wallet';
        console.log(img)
        const activation = net.infer(img, 'conv_preds');
        await classifier.addExample(activation, classId);
      }
    }

    if (classId === "banana") {
      
      for (let i = 0; i < bananaPhotoArr.length; i++) { 
        var img = new Image(250, 250); // Use DOM HTMLImageElement
        img.src =  bananaPhotoArr[i];
        img.alt = 'banana';
        console.log(img)
        const activation = net.infer(img, 'conv_preds');
        await classifier.addExample(activation, classId);
      }
    }
    
    console.log(classId, "Image feed done!");
    // document.body.appendChild(img);
    // const imgPixel = await tf.browser.fromPixels(img);

    // console.log(imgPixel)
    
  }


  const addExample = classId => {
    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.

    console.log(webcamElement)
    const activation = net.infer(webcamElement, 'conv_preds');

    console.log(`${classId} => ${activation}`)

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);
  };

  // document.getElementById('class-a').addEventListener('click', () => addExample("Apple"));
  // document.getElementById('class-b').addEventListener('click', () => addExample("banana"));

  document.getElementById('image-add').addEventListener('click', () => addImage("wallet"));
  document.getElementById('banana-image-add').addEventListener('click', () => addImage("banana"));


 
  document.getElementById('save-data').addEventListener('click', () => saveData());
  document.getElementById('load-data').addEventListener('click', () => loadData());

  while (true) {
    if (classifier.getNumClasses() > 0) {
      // Get the activation from mobilenet from the webcam.

      let img = new Image(250, 250);
      img.crossOrigin = "anonymous";
      img.src = "https://cdn.shopify.com/s/files/1/1078/0310/products/fruit-banana-dole-1_1024x1024.jpg?v=1500709708";
      img.alt = "banana";
      const activation = net.infer(img, 'conv_preds');
      // Get the most likely class and confidences from the classifier module.
      const result = await classifier.predictClass(activation);

      console.log(result)
    
      document.getElementById('console').innerText = `
        
        prediction: ${result.label}\n
        probability: ${result.confidences[result.label]}\n
      `;

    }

    // Give some breathing room by waiting for the next animation frame to
    // fire.
    await tf.nextFrame();
  }
}

app();
