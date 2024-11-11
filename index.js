const webcamEl = document.querySelector("#webcam");
const liveView = document.querySelector("#liveView");
const appSection = document.querySelector("#app");
const btnEnableWebcam = document.querySelector("#btnEnableWebcam");
const outputMessageEl = document.querySelector("#outputMessage");

let webcam;
let model;
let detectionHistory = [];
let knifePosition = null;
let knifeDetectionTime = null;

const confidenceThreshold = 0.66;
const maxHistoryLength = 5;
const knifeReminderThreshold = 30000; // 1 minuto en milisegundos

const kitchenObjects = [
  'bottle',
  'wine glass',
  'cup',
  'fork',
  'knife',
  'spoon',
  'bowl',
  'banana',
  'apple',
  'sandwich',
  'orange',
  'broccoli',
  'carrot',
  'hot dog',
  'pizza',
  'donut',
  'cake',
  'microwave',
  'oven',
  'toaster',
  'sink',
  'refrigerator',
  'dining table',
  'cell phone',
  'person'
];

function initTFJS() {
  if (typeof tf === "undefined") {
    throw new Error("TensorFlow.js not loaded");
  }
}

function getUserMediaSupported() {
  return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

async function enableCam(event) {
  if (!model) return;

  event.target.classList.add("removed");
  try {
    webcam = await tf.data.webcam(webcamEl);
    outputMessageEl.innerText = "Webcam enabled! Detecting objects...";
    await predictWebcam();
  } catch (error) {
    console.error("Webcam access failed:", error);
    outputMessageEl.innerText = "Webcam access denied.";
  }
}

async function loadCocoSsdModel() {
  model = await cocoSsd.load();
  btnEnableWebcam.disabled = false;
  outputMessageEl.innerText = "Model loaded! Enable the webcam to start detection.";
}

async function predictWebcam() {
  const objects = [];

  while (true) {
    const frame = await webcam.capture();
    const predictions = await model.detect(frame);

    objects.forEach((object) => liveView.removeChild(object));
    objects.length = 0;

    let foundBottle = false;
    let foundCup = false;

    for (let n = 0; n < predictions.length; n++) {
      if (predictions[n].score > confidenceThreshold && kitchenObjects.includes(predictions[n].class)) {
        const c = predictions[n].class;
        const score = Math.round(parseFloat(predictions[n].score) * 100);

        trackObjectPosition(c, predictions, n)

        // Comprobar si se encuentran "bottle" o "cup"
        if (c === 'person') foundBottle = true;
        if (c === 'cell phone') foundCup = true;

        const p = document.createElement("p");
        p.innerText = `${c} - with ${score}% confidence.`;
        p.style = `margin-left: ${predictions[n].bbox[0]}px; 
                   margin-top: ${predictions[n].bbox[1] - 10}px; 
                   width: ${predictions[n].bbox[2] - 10}px; 
                   top: 0; 
                   left: 0;`;

        const highlighter = document.createElement("div");
        highlighter.setAttribute("class", "highlighter");
        highlighter.style = `left: ${predictions[n].bbox[0]}px; 
                             top: ${predictions[n].bbox[1]}px; 
                             width: ${predictions[n].bbox[2]}px; 
                             height: ${predictions[n].bbox[3]}px;`;

        liveView.appendChild(highlighter);
        liveView.appendChild(p);
        objects.push(highlighter);
        objects.push(p);
      }
    }

    // Mostrar mensaje si se encuentran ambos objetos
    if (foundBottle && foundCup) {
      outputMessageEl.innerText = "¿Te gustaría tomar algo?";
    } else {
      outputMessageEl.innerText = "Detectando objetos...";
    }

    frame.dispose();
    await tf.nextFrame();
  }
}

function trackObjectPosition(c, predictions, n) {
  if (c === 'cell phone') {
    const knifeCurrentPosition = predictions[n].bbox;

    if (knifePosition && arePositionsEqual(knifePosition, knifeCurrentPosition)) {
      if (!knifeDetectionTime) {
        knifeDetectionTime = Date.now();
      } else if (Date.now() - knifeDetectionTime > knifeReminderThreshold) {
        displayKnifeReminder();
      }
    } else {
      knifePosition = knifeCurrentPosition;
      knifeDetectionTime = null;
    }
  }
}

function arePositionsEqual(pos1, pos2) {
  const tolerance = 20;
  return (
    Math.abs(pos1[0] - pos2[0]) < tolerance &&
    Math.abs(pos1[1] - pos2[1]) < tolerance &&
    Math.abs(pos1[2] - pos2[2]) < tolerance &&
    Math.abs(pos1[3] - pos2[3]) < tolerance
  );
}

function displayKnifeReminder() {
  outputMessageEl.innerText = "Recuerda guardar el cuchillo después de usarlo.";
}

async function app() {
  if (!getUserMediaSupported()) {
    console.warn("getUserMedia() is not supported by your browser");
    outputMessageEl.innerText = "Webcam not supported by your browser.";
    return;
  }
  btnEnableWebcam.addEventListener("click", enableCam);

  await loadCocoSsdModel();
}

(async function initApp() {
  try {
    initTFJS();
    await app();
  } catch (error) {
    console.error(error);
    outputMessageEl.innerText = error.message;
  }
}());
